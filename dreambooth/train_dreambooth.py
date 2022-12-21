import itertools
import logging
import math
import os
import random
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as f
import torch.utils.checkpoint
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import logging as dl
from huggingface_hub import HfFolder, whoami
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel

from extensions.sd_dreambooth_extension.dreambooth import xattention, dream_state
from extensions.sd_dreambooth_extension.dreambooth.SuperDataset import SuperDataset
from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig, from_file
from extensions.sd_dreambooth_extension.dreambooth.diff_to_sd import compile_checkpoint
from extensions.sd_dreambooth_extension.scripts.dreambooth import printm
from extensions.sd_dreambooth_extension.dreambooth.finetune_utils import encode_hidden_state, \
    EMAModel, generate_classifiers
from extensions.sd_dreambooth_extension.dreambooth.utils import cleanup, unload_system_models
from extensions.sd_dreambooth_extension.lora_diffusion.lora import save_lora_weight, apply_lora_weights
from modules import shared, paths, images

try:
    cmd_dreambooth_models_path = shared.cmd_opts.dreambooth_models_path
except:
    cmd_dreambooth_models_path = None

try:
    profile_memory = shared.cmd_opts.profile_db
except:
    profile_memory = False

if profile_memory:
    from torch.profiler import profile

mem_record = {}
with_prior = False

torch.backends.cudnn.benchmark = not profile_memory

logger = logging.getLogger(__name__)
# define a Handler which writes DEBUG messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.setLevel(logging.DEBUG)
dl.set_verbosity_error()

last_img_step = -1
last_save_step = -1
last_check_step = 0
last_check_epoch = 0


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache, concepts_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache
        self.concepts_cache = concepts_cache
        self.current_index = 0
        self.current_concept = 0

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        self.current_concept = self.concepts_cache[index]
        return self.latents_cache[index], self.text_encoder_cache[index]


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.avg: torch.Tensor = None
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args: DreamboothConfig, memory_record, use_subdir, lora_model=None, lora_alpha=1.0, lora_txt_alpha=1.0,
         custom_model_name="", use_txt2img=True) -> tuple[DreamboothConfig, dict, str]:
    logging_dir = Path(args.model_dir, "logging")
    if profile_memory:
        cleanup(True)
        prof = profile(
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=10),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{logging_dir}/dreambooth'),
            profile_memory=True)

        prof.start()
    else:
        prof = None

    global with_prior
    global mem_record

    text_encoder = None
    args.tokenizer_name = None
    mem_record = memory_record
    max_train_steps = args.max_train_steps

    args.max_token_length = int(args.max_token_length)
    if not args.pad_tokens and args.max_token_length > 75:
        print("Cannot raise token length limit above 75 when pad_tokens=False")

    if args.attention == "xformers":
        xattention.replace_unet_cross_attn_to_xformers()
    elif args.attention == "flash_attention":
        xattention.replace_unet_cross_attn_to_flash_attention()
    else:
        xattention.replace_unet_cross_attn_to_default()

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    try:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with="tensorboard",
            logging_dir=logging_dir,
            cpu=args.use_cpu
        )
    except Exception as e:
        if "AcceleratorState" in str(e):
            msg = "Change in precision detected, please restart the webUI entirely to use new precision."
        else:
            msg = f"Exception initializing accelerator: {e}"
        print(msg)
        return args, mem_record, msg
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        msg = "Gradient accumulation is not supported when training the text encoder in distributed training. " \
              "Please set gradient_accumulation_steps to 1. This feature will be supported in the future. Text " \
              "encoder training will be disabled."
        print(msg)
        dream_state.status.textinfo = msg
        args.train_text_encoder = False

    count, with_prior, _ = generate_classifiers(args, lora_model, lora_weight=lora_alpha, lora_text_weight=lora_txt_alpha,
                                                use_txt2img=use_txt2img, accelerator=accelerator)
    if use_txt2img and count > 0:
        print("Unloading system models (again).")
        unload_system_models()

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, "tokenizer"),
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load models and create wrapper for stable diffusion
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        torch_dtype=torch.float32
    )

    printm("Loaded model.")

    def create_vae():
        vae_path = args.pretrained_vae_name_or_path if args.pretrained_vae_name_or_path else \
            args.pretrained_model_name_or_path
        result = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder=None if args.pretrained_vae_name_or_path else "vae",
            revision=args.revision
        )
        result.requires_grad_(False)
        result.to(accelerator.device, dtype=weight_dtype)
        return result

    vae = create_vae()

    unet_lora_params = None
    text_encoder_lora_params = None

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.use_lora:
        unet_lora_params, text_encoder_lora_params = apply_lora_weights(lora_model, unet, text_encoder, lora_alpha,
                                                                        lora_txt_alpha, accelerator.device)

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size *
                accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    use_adam = False
    optimizer_class = torch.optim.AdamW

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            use_adam = True
        except Exception as a:
            logger.warning(f"Exception importing 8bit adam: {a}")
            traceback.print_exc()

    if args.use_lora:

        args.learning_rate = args.lora_learning_rate

        params_to_optimize = ([
                                  {"params": itertools.chain(*unet_lora_params), "lr": args.lora_learning_rate},
                                  {"params": itertools.chain(*text_encoder_lora_params),
                                   "lr": args.lora_txt_learning_rate},
                              ]
                              if args.train_text_encoder
                              else itertools.chain(*unet_lora_params)
                              )
    else:
        params_to_optimize = (
            itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder
            else unet.parameters()
        )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    def cleanup_memory():
        try:
            printm("CLEANUP: ")
            if unet:
                del unet
            if text_encoder:
                del text_encoder
            if tokenizer:
                del tokenizer
            if optimizer:
                del optimizer
            if train_dataloader:
                del train_dataloader
            if train_dataset:
                del train_dataset
            if lr_scheduler:
                del lr_scheduler
            if vae:
                del vae
            if ema_unet:
                del ema_unet
            if unet_lora_params:
                del unet_lora_params
        except:
            pass
        try:
            cleanup(True)
        except:
            pass
        printm("Cleanup Complete.")

    train_dataset = SuperDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        lifetime_steps=args.revision,
        pad_tokens=args.pad_tokens,
        hflip=args.hflip,
        max_token_length=args.max_token_length,
        shuffle_tags=args.shuffle_tags
    )

    if train_dataset.__len__ == 0:
        msg = "Please provide a directory with actual images in it."
        print(msg)
        dream_state.status.textinfo = msg
        cleanup_memory()
        return args, mem_record, msg

    def collate_fn(examples):
        input_ids = [ex["instance_prompt_ids"] for ex in examples]
        pixel_values = [ex["instance_images"] for ex in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if with_prior:
            input_ids += [ex["class_prompt_ids"] for ex in examples]
            pixel_values += [ex["class_images"] for ex in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        if not args.pad_tokens:
            input_ids = tokenizer.pad(
                {"input_ids": input_ids},
                padding=True,
                return_tensors="pt",
            ).input_ids
        else:
            input_ids = torch.stack(input_ids)

        output = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return output

    train_dataloader = torch.utils.data.DataLoader(
        # train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=1
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True
    )
    # Move text_encoder and VAE to GPU.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.

    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    def cache_latents(td=None, tdl=None, enc_vae=None, orig_dataset=None):
        global with_prior
        if td is not None:
            del td
        if tdl is not None:
            del tdl

        if enc_vae is None:
            enc_vae = create_vae()

        if orig_dataset is None:
            dataset = SuperDataset(
                concepts_list=args.concepts_list,
                tokenizer=tokenizer,
                size=args.resolution,
                center_crop=args.center_crop,
                lifetime_steps=args.revision,
                pad_tokens=args.pad_tokens,
                hflip=args.hflip,
                max_token_length=args.max_token_length,
                shuffle_tags=args.shuffle_tags
            )
        else:
            dataset = orig_dataset

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True
        )
        latents_cache = []
        text_encoder_cache = []
        concepts_cache = []
        for d_batch in tqdm(dataloader, desc="Caching latents", disable=True):
            c_concept = args.concepts_list[dataset.current_concept]
            with_prior = c_concept.num_class_images > 0
            with torch.no_grad():
                d_batch["pixel_values"] = d_batch["pixel_values"].to(accelerator.device, non_blocking=True,
                                                                     dtype=weight_dtype)
                d_batch["input_ids"] = d_batch["input_ids"].to(accelerator.device, non_blocking=True)
                latents_cache.append(enc_vae.encode(d_batch["pixel_values"]).latent_dist)
                if args.train_text_encoder:
                    text_encoder_cache.append(d_batch["input_ids"])
                else:
                    text_encoder_cache.append(text_encoder(d_batch["input_ids"])[0])
                concepts_cache.append(dataset.current_concept)
        dataset = LatentsDataset(latents_cache, text_encoder_cache, concepts_cache)
        dataloader = accelerator.prepare(
            torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=lambda z: z, shuffle=True))
        if enc_vae is not None:
            del enc_vae
        return dataset, dataloader

    # Store our original uncached dataset for preview generation
    gen_dataset = train_dataset

    if not args.not_cache_latents:
        train_dataset, train_dataloader = cache_latents(enc_vae=vae, orig_dataset=gen_dataset)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    # create ema, fix OOM
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())
        ema_unet.to(accelerator.device, dtype=weight_dtype)
        if args.train_text_encoder and text_encoder is not None:
            unet, ema_unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, ema_unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, ema_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, ema_unet, optimizer, train_dataloader, lr_scheduler
            )
    else:
        ema_unet = None
        if args.train_text_encoder and text_encoder is not None:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

    printm("Scheduler, EMA Loaded.")
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if max_train_steps is None or max_train_steps < 1:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    actual_train_steps = max_train_steps * args.train_batch_size * args.gradient_accumulation_steps
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    global_step = 0
    global_epoch = 0
    last_save_step = 0
    last_save_epoch = 0
    last_img_step = 0
    last_img_epoch = 0
    first_epoch = 0
    resume_step = 0
    resume_from_checkpoint = False
    new_hotness = os.path.join(args.model_dir, "checkpoints", f"checkpoint-{args.revision}")
    if os.path.exists(new_hotness):
        accelerator.print(f"Resuming from checkpoint {new_hotness}")
        try:
            no_safe = shared.cmd_opts.disable_safe_unpickle
        except:
            no_safe = False
        shared.cmd_opts.disable_safe_unpickle = True
        accelerator.load_state(new_hotness)
        shared.cmd_opts.disable_safe_unpickle = no_safe
        global_step = args.revision
        resume_from_checkpoint = True
        resume_global_step = global_step
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

    print("  ***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    print(f"  Resuming from checkpoint: {resume_from_checkpoint}")
    print(f"  First resume epoch: {first_epoch}")
    print(f"  First resume step: {resume_step}")
    print(f"  CPU: {args.use_cpu} Adam: {use_adam}, Prec: {args.mixed_precision}")
    print(f"  Grad: {args.gradient_checkpointing}, TextTr: {args.train_text_encoder} Use EMA: {args.use_ema}")
    print(f"  Learning rate: {args.learning_rate} Use lora:{args.use_lora}")

    last_img_step = -1
    last_save_step = -1
    last_check_step = 0
    last_check_epoch = 0

    def check_save():
        global last_save_step
        global last_img_step
        global last_check_step
        global last_check_epoch

        training_save_interval = args.save_embedding_every
        training_image_interval = args.save_preview_every

        if args.save_use_epochs:
            if args.save_use_global_counts:
                training_save_check = global_epoch
            else:
                training_save_check = args.epoch
        else:
            if args.save_use_global_counts:
                training_save_check = global_step
            else:
                training_save_check = args.revision

        save_completed = global_step >= actual_train_steps
        save_canceled = dream_state.status.interrupted
        save_image = False
        save_model = False
        if not save_canceled and not save_completed:
            if last_save_step == -1:
                save_model = False
                last_save_step = 0
            else:
                if training_save_check - last_save_step >= training_save_interval:
                    save_model = True
                    last_save_step = training_save_check
            if last_img_step == -1:
                save_image = False
                last_img_step = 0
            else:
                if training_save_check - last_img_step >= training_image_interval:
                    save_image = True
                    last_img_step = training_save_check

        else:
            print("Save completed/canceled.")
            save_image = True
            save_model = True

        save_snapshot = False
        save_lora = False
        save_checkpoint = False

        if save_model:
            if save_canceled:
                save_lora = args.save_lora_cancel
                save_snapshot = args.save_state_during
                save_checkpoint = args.save_ckpt_cancel
            elif save_completed:
                save_lora = args.save_lora_after
                save_snapshot = args.save_state_after
                save_checkpoint = args.save_ckpt_after
            else:
                save_lora = args.save_lora_during
                save_snapshot = args.save_state_during
                save_checkpoint = args.save_ckpt_during

        if save_checkpoint or save_snapshot or save_lora or save_image or save_model:
            print(f"Save fired {save_lora}, {save_checkpoint}, {save_snapshot}, {save_image}")
            save_weights(save_image, save_model, save_snapshot, save_checkpoint, save_lora)

        return save_model

    def save_weights(save_image, save_model, save_snapshot, save_checkpoint, save_lora):
        # Create the pipeline using the trained modules and save it.
        if accelerator.is_main_process:
            accelerator.save_state(os.path.join(args.model_dir, "checkpoints", f"checkpoint-{args.revision}"))
            g_cuda = None
            if args.train_text_encoder:
                text_enc_model = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
            else:
                text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                               subfolder="text_encoder",
                                                               revision=args.revision)
            pred_type = "epsilon"
            if args.v2:
                pred_type = "v_prediction"
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", steps_offset=1,
                                      clip_sample=False, set_alpha_to_one=False, prediction_type=pred_type)
            if args.use_ema:
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())

            s_pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet, keep_fp32_wrapper=True),
                text_encoder=text_enc_model,
                vae=vae if vae is not None else create_vae(),
                scheduler=scheduler,
                torch_dtype=torch.float16,
                revision=args.revision,
                safety_checker=None,
                requires_safety_checker=None
            )

            s_pipeline = s_pipeline.to(accelerator.device)

            with accelerator.autocast(), torch.inference_mode():
                if save_model:
                    try:
                        if args.use_lora and save_lora:
                            lora_model_name = args.model_name if custom_model_name == "" else custom_model_name
                            try:
                                cmd_lora_models_path = shared.cmd_opts.lora_models_path
                            except:
                                cmd_lora_models_path = None
                            model_dir = os.path.dirname(
                                cmd_lora_models_path) if cmd_lora_models_path else paths.models_path
                            out_file = os.path.join(model_dir, "lora")
                            os.makedirs(out_file, exist_ok=True)
                            out_file = os.path.join(out_file, f"{lora_model_name}_{args.revision}.pt")
                            print(f"\nSaving lora weights at step {args.revision}")
                            # Save a pt file
                            save_lora_weight(s_pipeline.unet, out_file)
                            if args.train_text_encoder:
                                out_txt = out_file.replace(".pt", "_txt.pt")
                                save_lora_weight(s_pipeline.text_encoder,
                                                 out_txt,
                                                 target_replace_module=["CLIPAttention"],
                                                 )
                            print(f"\nLora weights successfully saved to {out_file}")
                        else:
                            out_file = None
                            if save_snapshot:
                                dream_state.status.textinfo = f"Saving snapshot at step {args.revision}..."
                                print("Saving snapshot.")
                                accelerator.save_state(os.path.join(args.model_dir, "checkpoints",
                                                                    f"checkpoint-{args.revision}"))
                            else:
                                dream_state.status.textinfo = f"Saving diffusion model at step {args.revision}..."
                                print("Saving diffusion models.")
                                s_pipeline.save_pretrained(os.path.join(args.model_dir, "working"))

                        if save_checkpoint:
                            compile_checkpoint(args.model_name, half=args.half_model, use_subdir=use_subdir,
                                               reload_models=False, lora_path=out_file, log=False,
                                               custom_model_name=custom_model_name)
                        if args.use_ema:
                            ema_unet.restore(unet.parameters())
                        args.save()
                    except Exception as ex:
                        print(f"Exception saving checkpoint/model: {ex}")
                        traceback.print_exc()
                        pass
                save_dir = args.model_dir
                if save_image:
                    dream_state.status.textinfo = f"Saving preview image at step {args.revision}..."
                    try:
                        s_pipeline.set_progress_bar_config(disable=True)
                        sample_dir = os.path.join(save_dir, "samples")
                        os.makedirs(sample_dir, exist_ok=True)
                        with accelerator.autocast(), torch.inference_mode():
                            prompts = gen_dataset.get_sample_prompts()
                            ci = 0
                            samples = []
                            for c in prompts:
                                seed = c.seed
                                if seed is None or seed == '' or seed == -1:
                                    seed = int(random.randrange(21474836147))
                                g_cuda = torch.Generator(device=accelerator.device).manual_seed(seed)
                                for si in tqdm(range(c.n_samples), desc="Generating samples"):
                                    s_image = s_pipeline(c.prompt, num_inference_steps=c.steps,
                                                         guidance_scale=c.scale,
                                                         negative_prompt=c.negative_prompt,
                                                         height=args.resolution,
                                                         width=args.resolution,
                                                         generator=g_cuda).images[0]
                                    dream_state.status.current_image = s_image
                                    samples.append(s_image)
                                    image_name = os.path.join(sample_dir, f"sample_{args.revision}-{ci}{si}.png")
                                    txt_name = image_name.replace(".jpg", ".txt")
                                    with open(txt_name, "w", encoding="utf8") as txt_file:
                                        txt_file.write(c.prompt)
                                    s_image.save(image_name)
                                ci += 1
                            if len(samples) > 1:
                                img_grid = images.image_grid(samples)
                                dream_state.status.current_image = img_grid
                                del samples

                    except Exception as em:
                        print(f"Exception with the stupid image again: {em}")
                        traceback.print_exc()
                        pass

            del s_pipeline
            del scheduler
            del text_enc_model
            if save_image:
                if g_cuda:
                    del g_cuda

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    lifetime_step = args.revision
    dream_state.status.job_count = max_train_steps
    dream_state.status.job_no = global_step
    dream_state.status.textinfo = f"Training step: {global_step}/{max_train_steps}"
    loss_avg = AverageMeter()
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()
    training_complete = False
    msg = ""
    weights_saved = False
    for epoch in range(first_epoch, args.num_train_epochs):
        if training_complete:
            break
        try:
            unet.train()
            if args.train_text_encoder and text_encoder is not None:
                text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue
                with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):
                    # Convert images to latent space
                    with torch.no_grad():
                        if not args.not_cache_latents:
                            latent_dist = batch[0][0]
                        else:
                            latent_dist = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist
                        latents = latent_dist.sample() * 0.18215
                        b_size = latents.shape[0]

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                              device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    with text_enc_context:
                        if not args.not_cache_latents:
                            if args.train_text_encoder:
                                encoder_hidden_states = encode_hidden_state(text_encoder, batch[0][1], args.pad_tokens,
                                                                            b_size, args.max_token_length,
                                                                            tokenizer.model_max_length)
                            else:
                                encoder_hidden_states = batch[0][1]
                        else:
                            encoder_hidden_states = encode_hidden_state(text_encoder, batch["input_ids"],
                                                                        args.pad_tokens, b_size, args.max_token_length,
                                                                        tokenizer.model_max_length)

                    # Predict the noise residual
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        noise = noise_scheduler.get_velocity(latents, noise, timesteps)

                    concept_index = train_dataset.current_concept
                    concept = args.concepts_list[concept_index]
                    if concept.num_class_images > 0:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        noise, noise_prior = torch.chunk(noise, 2, dim=0)

                        # Compute instance loss
                        loss = f.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

                        # Compute prior loss
                        prior_loss = f.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                    else:
                        loss = f.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if profile_memory:
                        loss_avg.update(loss.cpu().float(), bsz)
                    else:
                        loss_avg.update(loss.detach_(), bsz)

                    # Update EMA
                    if args.use_ema and ema_unet is not None:
                        ema_unet.step(unet.parameters())

                if not global_step % 2:
                    allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
                    cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
                    logs = {"loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0],
                            "vram": f"{allocated}/{cached}GB"}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=args.revision)
                    loss_avg.reset()

                actual_steps = args.train_batch_size * args.gradient_accumulation_steps
                progress_bar.update(actual_steps)
                global_step += actual_steps
                args.revision += actual_steps
                dream_state.status.job_no = global_step

                # Check training complete before save check
                training_complete = global_step >= actual_train_steps or dream_state.status.interrupted

                weights_saved = check_save()

                # Reset the job count after saving images
                dream_state.status.job_count = actual_train_steps

                # Check again after possibly saving
                if dream_state.status.interrupted:
                    training_complete = True

                dream_state.status.textinfo = f"Training, step {global_step}/{actual_train_steps} current," \
                                        f" {args.revision}/{actual_train_steps + lifetime_step} lifetime"

                # Log completion message
                if training_complete:
                    print("  Training complete.")
                    if dream_state.status.interrupted:
                        state = "cancelled"
                    else:
                        state = "complete"

                    dream_state.status.textinfo = f"Training {state} {global_step}/{actual_train_steps}, {args.revision}" \
                                            f" total."

                    break

            # Check after epoch
            training_complete = global_step >= actual_train_steps or dream_state.status.interrupted
            accelerator.wait_for_everyone()

            if not args.not_cache_latents:
                train_dataset, train_dataloader = cache_latents(enc_vae=vae, orig_dataset=gen_dataset)

            if training_complete:
                if profile_memory and prof is not None:
                    prof.step()

                msg = f"Training completed, total steps: {args.revision}"
                break
        except Exception as m:
            msg = f"Exception while training: {m}"
            printm(msg)
            traceback.print_exc()
            mem_summary = torch.cuda.memory_summary()
            print(mem_summary)
            break
        if dream_state.status.interrupted:
            training_complete = True

        args.epoch += 1
        global_epoch += 1
        args.save()
        if not weights_saved:
            check_save()
        dream_state.status.job_count = actual_train_steps

        if args.epoch_pause_frequency > 0 and args.epoch_pause_time > 0:
            if not global_epoch % args.epoch_pause_frequency:
                print(f"Giving the GPU a break for {args.epoch_pause_time} seconds.")
                for i in range(args.epoch_pause_time):
                    if dream_state.status.interrupted:
                        training_complete = True
                        break
                    time.sleep(1)

        if training_complete:
            break

    if profile_memory and prof is not None:
        prof.stop()
    cleanup_memory()
    accelerator.end_training()
    return args, mem_record, msg
