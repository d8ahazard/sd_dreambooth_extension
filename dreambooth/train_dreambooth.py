# Borrowed heavily from https://github.com/bmaltais/kohya_ss/blob/master/train_db.py and
# https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth
# With some custom bits sprinkled in and some stuff from OG diffusers as well.

import gc
import itertools
import logging
import os
import random
import time
import traceback
from decimal import Decimal
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from diffusers.utils import logging as dl
from huggingface_hub import HfFolder, whoami
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from extensions.sd_dreambooth_extension.dreambooth import xattention, db_shared
from extensions.sd_dreambooth_extension.dreambooth.SuperDataset import SampleData
from extensions.sd_dreambooth_extension.dreambooth.db_concept import Concept
from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig
from extensions.sd_dreambooth_extension.dreambooth.db_optimization import get_scheduler
from extensions.sd_dreambooth_extension.dreambooth.db_shared import status
from extensions.sd_dreambooth_extension.dreambooth.diff_to_sd import compile_checkpoint
from extensions.sd_dreambooth_extension.dreambooth.finetune_utils import EMAModel, generate_classifiers, \
    FilenameTextGetter, PromptData
from extensions.sd_dreambooth_extension.dreambooth.finetuneing_dataset_2 import DreamBoothOrFineTuningDataset
from extensions.sd_dreambooth_extension.dreambooth.memory import find_executable_batch_size
from extensions.sd_dreambooth_extension.dreambooth.sample_dataset import SampleDataset
from extensions.sd_dreambooth_extension.dreambooth.utils import cleanup, unload_system_models, parse_logs, get_images, \
    printm
from extensions.sd_dreambooth_extension.lora_diffusion.lora import save_lora_weight, apply_lora_weights
from modules import shared, paths

try:
    cmd_dreambooth_models_path = shared.cmd_opts.dreambooth_models_path
except Exception:
    cmd_dreambooth_models_path = None

try:
    profile_memory = db_shared.profile_db
except Exception:
    profile_memory = False

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
        self.avg: float = 0
        self.count = 0
        self.counts = []

    def reset(self):
        self.count = self.avg = 0
        self.counts = []

    def update(self, val, n=1):
        self.counts.append(val * n)
        if len(self.counts) > 10:
            self.counts.pop(0)
        self.avg = float(sum(self.counts) / len(self.counts))


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


last_samples = []
last_prompts = []


class TrainResult:
    config: DreamboothConfig = None
    msg: str = ""
    samples: [Image] = []


def set_diffusers_xformers_flag(model, valid):
    # Recursively walk through all the children.
    # Any children which exposes the set_use_memory_efficient_attention_xformers method
    # gets the message
    def fn_recursive_set_mem_eff(module: torch.nn.Module):
        if hasattr(module, 'set_use_memory_efficient_attention_xformers'):
            module.set_use_memory_efficient_attention_xformers(valid)

        for child in module.children():
            fn_recursive_set_mem_eff(child)

    fn_recursive_set_mem_eff(model)


def collate_fn(examples):
    return examples[0]


def main(args: DreamboothConfig, use_subdir, lora_model=None, lora_alpha=1.0, lora_txt_alpha=1.0,
         custom_model_name="", use_txt2img=True) -> TrainResult:
    """

    @param args: The model config to use. 
    @param use_subdir: Save checkpoints to a subdirectory.
    @param lora_model: An optional lora model to use/resume.
    @param lora_alpha: The weight to use when applying lora unet.
    @param lora_txt_alpha: The weight to use when applying lora text encoder.
    @param custom_model_name: A custom name to use when saving checkpoints.
    @param use_txt2img: Use txt2img when generating class images.
    @return: TrainResult
    """
    logging_dir = Path(args.model_dir, "logging")

    result = TrainResult
    result.config = args

    @find_executable_batch_size(starting_batch_size=args.train_batch_size,
                                starting_grad_size=args.gradient_accumulation_steps)
    def inner_loop(train_batch_size, gradient_accumulation_steps):
        text_encoder = None
        args.tokenizer_name = None
        global last_samples
        global last_prompts
        global profile_memory
        if profile_memory:
            from torch.profiler import profile

            cleanup(True)

            prof = profile(
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=10),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{logging_dir}'),
                profile_memory=True)

            prof.start()
        else:
            prof = None

        n_workers = min(8, os.cpu_count() - 1)
        if os.name == "nt":
            n_workers = 0
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
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=args.mixed_precision,
                log_with="tensorboard",
                logging_dir=logging_dir
            )
        except Exception as e:
            if "AcceleratorState" in str(e):
                msg = "Change in precision detected, please restart the webUI entirely to use new precision."
            else:
                msg = f"Exception initializing accelerator: {e}"
            print(msg)
            result.msg = msg
            result.config = args
            return result
        # Currently, it's not possible to do gradient accumulation when training two models with
        # accelerate.accumulate This will be enabled soon in accelerate. For now, we don't allow gradient
        # accumulation when training two models.
        # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
        if args.stop_text_encoder != 0 and gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
            msg = "Gradient accumulation is not supported when training the text encoder in distributed training. " \
                  "Please set gradient_accumulation_steps to 1. This feature will be supported in the future. Text " \
                  "encoder training will be disabled."
            print(msg)
            status.textinfo = msg
            args.stop_text_encoder = 0

        count, _, _ = generate_classifiers(args, lora_model, lora_weight=lora_alpha,
                                           lora_text_weight=lora_txt_alpha,
                                           use_txt2img=use_txt2img, accelerator=accelerator)
        if use_txt2img and count > 0:
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

        if args.attention == "xformers":
            # print("Setting diffusers unet flags for xformers.")
            set_diffusers_xformers_flag(unet, True)

        vae_path = args.pretrained_vae_name_or_path if args.pretrained_vae_name_or_path else \
            args.pretrained_model_name_or_path
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder=None if args.pretrained_vae_name_or_path else "vae",
            revision=args.revision
        )
        vae.requires_grad_(False)
        vae.to(accelerator.device, dtype=weight_dtype)

        unet_lora_params = None
        text_encoder_lora_params = None

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            text_encoder.gradient_checkpointing_enable()

        if args.use_lora:
            unet_lora_params, text_encoder_lora_params = apply_lora_weights(lora_model, unet, text_encoder, lora_alpha,
                                                                            lora_txt_alpha, accelerator.device)

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

        # noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        #                                 num_train_timesteps=1000, clip_sample=False)
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

        def cleanup_memory():
            try:
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

        def load_dreambooth_dir(db_dir, concept: Concept, is_class: bool = True):
            img_paths = get_images(db_dir)
            captions = []
            text_getter = FilenameTextGetter()
            for img_path in img_paths:
                cap_for_img = text_getter.read_text(img_path)
                final_caption = text_getter.create_text(concept.instance_prompt, cap_for_img, concept.instance_token,
                                                        concept.class_token, is_class)
                captions.append(final_caption)

            return list(zip(img_paths, captions))

        print("prepare train images.")
        train_img_path_captions = []
        reg_img_path_captions = []
        tokens = []
        for conc in args.concepts_list:
            if conc.class_token != "" and conc.instance_token != "":
                tokens.append((conc.instance_token, conc.class_token))
            idd = conc.instance_data_dir
            if idd is not None and idd != "" and os.path.exists(idd):
                img_caps = load_dreambooth_dir(idd, conc, False)
                train_img_path_captions.extend(img_caps)
                print(f"{len(train_img_path_captions)} train images with repeating.")

            class_data_dir = conc.class_data_dir
            number_class_images = conc.num_class_images_per
            if number_class_images > 0 and class_data_dir is not None and class_data_dir != "" and os.path.exists(class_data_dir):
                print(f"Preparing class images from dir {class_data_dir}...")
                reg_caps = load_dreambooth_dir(class_data_dir, conc)
                reg_img_path_captions.extend(reg_caps)
            print(f"{len(reg_img_path_captions)} reg images.")

        # TODO: Add UI stuff for these
        resolution = (args.resolution, args.resolution)
        enable_bucket = True
        debug_bucket = False
        min_bucket_reso = int(args.resolution * 0.28125)  # 16x9 / 2
        max_bucket_reso = args.resolution
        # Enable to crop faces? (2.0,4.0)
        face_crop_aug_range = None

        if len(resolution) == 1:
            resolution = (resolution[0], resolution[0])
        assert len(resolution) == 2, \
            f"resolution must be 'size' or 'width,height' / resolutionは'サイズ'または'幅','高さ'で指定してください: {args.resolution}"

        if face_crop_aug_range is not None and isinstance(face_crop_aug_range, str):
            face_crop_aug_range = tuple([float(r) for r in face_crop_aug_range.split(',')])
            assert len(
                face_crop_aug_range) == 2, f"face_crop_aug_range must be two floats / face_crop_aug_range: {face_crop_aug_range}"
        else:
            face_crop_aug_range = None

        print("Preparing dataset")
        train_dataset = DreamBoothOrFineTuningDataset(
            batch_size=train_batch_size,
            fine_tuning=False,
            train_img_path_captions=train_img_path_captions,
            reg_img_path_captions=reg_img_path_captions,
            tokens=tokens,
            tokenizer=tokenizer,
            resolution=resolution,
            prior_loss_weight=args.prior_loss_weight,
            flip_aug=args.hflip,
            color_aug=False,
            face_crop_aug_range=face_crop_aug_range,
            random_crop=False,
            shuffle_caption=args.shuffle_tags,
            disable_padding=not args.pad_tokens,
            debug_dataset=True
        )

        if debug_bucket:
            train_dataset.make_buckets_with_caching(enable_bucket, None, min_bucket_reso, max_bucket_reso)
            print(f"Total dataset length (steps): {len(train_dataset)}")
            print("Escape for exit.")
            for example in train_dataset:
                k = None
                for im, cap, lw in zip(example['images'], example['captions'], example['loss_weights']):
                    im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))  # c,H,W -> H,W,c
                    im = im[:, :, ::-1]  # RGB -> BGR (OpenCV)
                    print(f'size: {im.shape[1]}*{im.shape[0]}, caption: "{cap}", loss weight: {lw}')
                    cv2.imshow("img", im)
                    k = cv2.waitKey()
                    cv2.destroyAllWindows()
                    if k == 27:
                        break
                if k == 27:
                    break
            return

        if train_dataset.__len__ == 0:
            msg = "Please provide a directory with actual images in it."
            print(msg)
            status.textinfo = msg
            cleanup_memory()
            result.msg = msg
            result.config = args
            return result

        # if args.stop_text_encoder != 0:
        #     text_encoder.to(accelerator.device, dtype=weight_dtype)
        #

        if args.cache_latents:
            vae.to(accelerator.device, dtype=weight_dtype)
            vae.requires_grad_(False)
            vae.eval()
            with torch.no_grad():
                train_dataset.make_buckets_with_caching(enable_bucket, vae, min_bucket_reso, max_bucket_reso)
            vae.to('cpu')
        else:
            train_dataset.make_buckets_with_caching(enable_bucket, None, min_bucket_reso, max_bucket_reso)
            vae.requires_grad_(False)
            vae.eval()

        unet.requires_grad_(True)  # 念のため追加
        text_encoder.requires_grad_(True)

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            text_encoder.gradient_checkpointing_enable()
        if args.use_lora:
            params_to_optimize = ([
                {"params": itertools.chain(*unet_lora_params), "lr": args.lora_learning_rate},
                {"params": itertools.chain(*text_encoder_lora_params),"lr": args.lora_txt_learning_rate}
            ])
        else:
            params_to_optimize = (
                itertools.chain(unet.parameters(), text_encoder.parameters())
            )

        optimizer = optimizer_class(params_to_optimize, lr=args.learning_rate if not args.use_lora else args.lora_learning_rate)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=n_workers)

        max_train_steps = args.num_train_epochs * len(train_dataloader) * train_batch_size

        # This is separate, because optimizer.step is only called once per "step" in training, so it's not
        # affected by batch size
        sched_train_steps = args.num_train_epochs * train_dataset.num_train_images

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * gradient_accumulation_steps,
            total_training_steps=sched_train_steps,
            num_cycles=args.lr_cycles,
            power=args.lr_power,
            factor=args.lr_factor,
            scale_pos=args.lr_scale_pos,
            min_lr=args.learning_rate_min
        )

        # create ema, fix OOM
        if args.use_ema:
            ema_unet = EMAModel(unet.parameters())
            ema_unet.to(accelerator.device, dtype=weight_dtype)
            unet, ema_unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, ema_unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            ema_unet = None
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )

        if not args.cache_latents and vae is not None:
            vae.to(accelerator.device, dtype=weight_dtype)
        # Afterwards we recalculate our number of training epochs
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers will initialize automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("dreambooth")

        # Train!
        total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
        max_train_epochs = args.num_train_epochs
        # we calculate our number of tenc training epochs
        text_encoder_epochs=round(args.num_train_epochs*args.stop_text_encoder)
        global_step = 0
        global_epoch = 0
        last_save_step = 0
        last_img_step = 0
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
            try:
                shared.cmd_opts.disable_safe_unpickle = True
                accelerator.load_state(new_hotness)
                shared.cmd_opts.disable_safe_unpickle = no_safe
                global_step = args.revision
                resume_from_checkpoint = True
                resume_global_step = global_step
                first_epoch = args.epoch
                resume_step = resume_global_step
            except Exception as lex:
                print(f"Exception loading checkpoint: {lex}")

        print("  ***** Running training *****")
        print(f"  Instance Images: {train_dataset.num_train_images}")
        print(f"  Class Images: {train_dataset.num_reg_images}")
        print(f"  Total Examples: {train_dataset.num_train_images * (2 if train_dataset.enable_reg_images else 1)}")
        print(f"  Num batches each epoch = {len(train_dataloader)}")
        print(f"  Num Epochs = {max_train_epochs}")
        print(f"  Batch Size Per Device = {train_batch_size}")
        print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Text Encoder Epochs: {text_encoder_epochs}")
        print(f"  Total optimization steps = {sched_train_steps}")
        print(f"  Total training steps = {max_train_steps}")
        print(f"  Resuming from checkpoint: {resume_from_checkpoint}")
        print(f"  First resume epoch: {first_epoch}")
        print(f"  First resume step: {resume_step}")
        print(f"  Lora: {args.use_lora}, Adam: {use_adam}, Prec: {args.mixed_precision}")
        print(f"  Gradient Checkpointing: {args.gradient_checkpointing}")
        print(f"  EMA: {args.use_ema}")
        print(f"  LR: {args.learning_rate})")

        last_img_step = -1
        last_save_step = -1
        last_check_step = 0
        last_check_epoch = 0

        def optim_to(optim: torch.optim.Optimizer, device="cpu"):
            def inplace_move(obj: torch.Tensor, target):
                if hasattr(obj, 'data'):
                    obj.data = obj.data.to(target)
                if hasattr(obj, '_grad') and obj._grad is not None:
                    obj._grad.data = obj._grad.data.to(target)

            if isinstance(optim, torch.optim.Optimizer):
                for param in optim.state.values():
                    if isinstance(param, torch.Tensor):
                        inplace_move(param, device)
                    elif isinstance(param, dict):
                        for subparams in param.values():
                            inplace_move(subparams, device)
            torch.cuda.empty_cache()
        def check_save(pbar: tqdm, is_epoch_check = False):
            global last_save_step
            global last_img_step
            global last_check_step
            global last_check_epoch
            training_save_interval = args.save_embedding_every
            training_image_interval = args.save_preview_every
            training_completed_count = max_train_epochs
            training_save_check = global_epoch
            save_completed = training_save_check >= training_completed_count
            save_canceled = status.interrupted
            save_image = False
            save_model = False
            if not save_canceled and not save_completed:
                if last_save_step == -1:
                    save_model = False
                    last_save_step = 0
                else:
                    if training_save_interval == 0:
                        save_model = False
                    elif training_save_check - last_save_step >= training_save_interval:
                        save_model = True
                        last_save_step = training_save_check

                if last_img_step == -1:
                    save_image = False
                    last_img_step = 0
                else:
                    if training_image_interval == 0:
                        save_image = False
                    elif training_save_check - last_img_step >= training_image_interval:
                        save_image = True
                        last_img_step = training_save_check

            else:
                print("\nSave completed/canceled.")
                if global_step > 0:
                    save_image = True
                    save_model = True

            save_snapshot = False
            save_lora = False
            save_checkpoint = False
            if db_shared.status.do_save_samples and is_epoch_check:
                save_image = True
                db_shared.status.do_save_samples = False

            if db_shared.status.do_save_model and is_epoch_check:
                save_model = True
                db_shared.status.do_save_model = False

            if save_model:
                if save_canceled:
                    print("Canceled, enabling saves.")
                    save_lora = args.save_lora_cancel
                    save_snapshot = args.save_state_cancel
                    save_checkpoint = args.save_ckpt_cancel
                elif save_completed:
                    print("Completed, enabling saves.")
                    save_lora = args.save_lora_after
                    save_snapshot = args.save_state_after
                    save_checkpoint = args.save_ckpt_after
                else:
                    save_lora = args.save_lora_during
                    save_snapshot = args.save_state_during
                    save_checkpoint = args.save_ckpt_during

            if save_checkpoint or save_snapshot or save_lora or save_image or save_model:
                printm(" Saving weights.")
                save_weights(save_image, save_model, save_snapshot, save_checkpoint, save_lora, pbar)
                pbar.set_description("Steps")
                pbar.reset(max_train_steps)
                pbar.update(global_step)
                printm(" Complete.")
                cleanup()
                printm("Cleaned again.")

            return save_model

        def save_weights(save_image, save_model, save_snapshot, save_checkpoint, save_lora, pbar):
            global last_samples
            global last_prompts
            # Create the pipeline using the trained modules and save it.
            if accelerator.is_main_process:
                printm("Precleanup")
                optim_to(optimizer)
                gc.collect()
                cleanup()
                g_cuda = None
                pred_type = "epsilon"
                if args.v2:
                    pred_type = "v_prediction"
                printm("Loading scheduler")
                scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                          steps_offset=1, clip_sample=False, set_alpha_to_one=False,
                                          prediction_type=pred_type)
                if args.use_ema:
                    printm("Storing ema_unet params?")
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())

                if args.cache_latents:
                    printm("Moving vae to accelerator device.")
                    vae.to(accelerator.device, dtype=weight_dtype)

                printm("Creating pipeline.")
                s_pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet, keep_fp32_wrapper=True),
                    text_encoder=accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True),
                    vae=vae,
                    scheduler=scheduler,
                    torch_dtype=weight_dtype,
                    revision=args.revision,
                    safety_checker=None,
                    requires_safety_checker=None
                )
                printm("Pipeline created.")
                s_pipeline = s_pipeline.to(accelerator.device)
                printm("Pipeline moved to device.")
                s_pipeline.enable_attention_slicing()
                with accelerator.autocast(), torch.inference_mode():
                    if save_model:
                        # We are saving weights, we need to ensure revision is saved
                        args.save()
                        pbar.set_description("Saving weights")
                        pbar.reset(4)
                        pbar.update()
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
                                # print(f"\nSaving lora weights at step {args.revision}")
                                # Save a pt file
                                save_lora_weight(s_pipeline.unet, out_file)
                                if args.stop_text_encoder != 0:
                                    out_txt = out_file.replace(".pt", "_txt.pt")
                                    save_lora_weight(s_pipeline.text_encoder,
                                                     out_txt,
                                                     target_replace_module=["CLIPAttention"],
                                                     )
                                    pbar.update()
                            elif not args.use_lora:
                                out_file = None
                                if save_snapshot:
                                    status.textinfo = f"Saving snapshot at step {args.revision}..."
                                    # print(f"Saving snapshot at step: {args.revision}")
                                    accelerator.save_state(os.path.join(args.model_dir, "checkpoints",
                                                                        f"checkpoint-{args.revision}"))
                                    pbar.update()

                                # We should save this regardless, because it's our fallback if no snapshot exists.
                                status.textinfo = f"Saving diffusion model at step {args.revision}..."
                                # print(f"Saving diffusion weights at step: {args.revision}.")
                                s_pipeline.save_pretrained(os.path.join(args.model_dir, "working"))
                                pbar.update()

                            if save_checkpoint:
                                compile_checkpoint(args.model_name, half=args.half_model, use_subdir=use_subdir,
                                                   reload_models=False, lora_path=out_file, log=False,
                                                   custom_model_name=custom_model_name)
                                pbar.update()
                            if args.use_ema:
                                printm("Restoring ema unet.")
                                ema_unet.restore(unet.parameters())
                                ema_unet.to(accelerator.device, dtype=weight_dtype)
                                printm("Restored, moved to acc.device.")
                        except Exception as ex:
                            print(f"Exception saving checkpoint/model: {ex}")
                            traceback.print_exc()
                            pass
                    save_dir = args.model_dir
                    if save_image:
                        samples = []
                        sample_prompts = []
                        last_samples = []
                        last_prompts = []
                        status.textinfo = f"Saving preview image(s) at step {args.revision}..."
                        try:
                            s_pipeline.set_progress_bar_config(disable=True)
                            sample_dir = os.path.join(save_dir, "samples")
                            os.makedirs(sample_dir, exist_ok=True)
                            with accelerator.autocast(), torch.inference_mode():
                                sd = SampleDataset(args.concepts_list, args.shuffle_tags)
                                prompts = sd.get_prompts()
                                if args.sanity_prompt != "" and args.sanity_prompt is not None:
                                    epd = PromptData()
                                    epd.prompt = args.sanity_prompt
                                    epd.seed = args.sanity_seed
                                    epd.negative_prompt = args.concepts_list[0].save_sample_negative_prompt
                                    extra = SampleData(args.sanity_prompt, concept=args.concepts_list[0])
                                    extra.seed = args.sanity_seed
                                    prompts.append(extra)
                                pbar.set_description("Previews")
                                pbar.reset(len(prompts) + 2)
                                ci = 0
                                for c in prompts:
                                    seed = int(c.seed)
                                    if seed is None or seed == '' or seed == -1:
                                        seed = int(random.randrange(21474836147))
                                    g_cuda = torch.Generator(device=accelerator.device).manual_seed(seed)
                                    s_image = s_pipeline(c.prompt, num_inference_steps=c.steps,
                                                         guidance_scale=c.scale,
                                                         negative_prompt=c.negative_prompt,
                                                         height=args.resolution,
                                                         width=args.resolution,
                                                         generator=g_cuda).images[0]
                                    sample_prompts.append(c.prompt)
                                    samples.append(s_image)
                                    image_name = os.path.join(sample_dir, f"sample_{args.revision}-{ci}.png")
                                    txt_name = image_name.replace(".png", ".txt")
                                    with open(txt_name, "w", encoding="utf8") as txt_file:
                                        txt_file.write(c.prompt)
                                    s_image.save(image_name)
                                    pbar.update()
                                    ci += 1
                                for sample in samples:
                                    last_samples.append(sample)
                                for prompt in sample_prompts:
                                    last_prompts.append(prompt)
                                del samples

                        except Exception as em:
                            print(f"Exception saving sample: {em}")
                            traceback.print_exc()
                            pass
                printm("Starting cleanup.")
                del s_pipeline
                s_pipeline = None
                del scheduler
                scheduler = None
                if save_image:
                    if g_cuda:
                        del g_cuda
                        g_cuda = None
                    try:
                        log_images, log_names = parse_logs(model_name=args.model_name)
                        pbar.update()
                        for log_image in log_images:
                            last_samples.append(log_image)
                        for log_name in log_names:
                            last_prompts.append(log_name)
                        db_shared.status.sample_prompts = last_prompts
                        pbar.update()
                        del log_images
                    except:
                        pass
                if args.cache_latents:
                    printm("Moving vae to cpu.")
                    vae.to("cpu")
                if torch.has_cuda:
                    printm("Emptying cache.")
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                unload_system_models()
                status.current_image = last_samples
                printm("Cleanup.")
                optim_to(optimizer, accelerator.device)
                cleanup()
                printm("Cleanup completed.")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        lifetime_step = args.revision
        status.job_count = max_train_steps
        status.job_no = global_step
        training_complete = False
        msg = ""
        for epoch in range(first_epoch, max_train_epochs):
            weights_saved = False
            if training_complete:
                print("Training complete, breaking epoch.")
                break

            unet.train()
            train_tenc = epoch<text_encoder_epochs
            text_encoder.train(train_tenc)
            text_encoder.requires_grad_(train_tenc)

            loss_total = 0

            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue
                with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):
                    # Convert images to latent space
                    with torch.no_grad():
                        if args.cache_latents:
                            latents = batch["latents"].to(accelerator.device)
                        else:
                            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                        latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents, device=latents.device)
                    b_size = latents.shape[0]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_size,),
                                              device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    enc_out = text_encoder(batch["input_ids"], output_hidden_states=True, return_dict=True)
                    encoder_hidden_states = enc_out['hidden_states'][-int(args.clip_skip)]
                    # encoder_hidden_states = encoder_hidden_states.to(device=accelerator.device, dtype=weight_dtype)
                    encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

                    # Predict the noise residual
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean([1, 2, 3])

                    loss_weights = batch["loss_weights"]
                    loss = loss * loss_weights

                    loss = loss.mean()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (itertools.chain(unet.parameters(), text_encoder.parameters()))
                        accelerator.clip_grad_norm_(params_to_clip, 1.0)

                    optimizer.step()
                    lr_scheduler.step()
                    if profile_memory:
                        prof.step()

                    # Update EMA
                    if args.use_ema and ema_unet is not None:
                        ema_unet.step(unet.parameters())

                    optimizer.zero_grad(set_to_none=args.gradient_set_to_none)

                current_loss = loss.detach().item()
                loss_total += current_loss
                avg_loss = loss_total / (step + 1)

                allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
                cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
                last_lr = lr_scheduler.get_last_lr()[0]
                global_step += train_batch_size
                args.revision += train_batch_size
                status.job_no += train_batch_size
                del noise_pred
                del latents
                del encoder_hidden_states
                del noise
                del timesteps
                del noisy_latents
                del target
                if torch.has_cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                logs = {"loss": float(current_loss), "loss_avg": avg_loss, "lr": last_lr, "vram_usage": float(cached)}
                status.textinfo2 = f"Loss: {'%.2f' % current_loss}, LR: {'{:.2E}'.format(Decimal(last_lr))}, " \
                                   f"VRAM: {allocated}/{cached} GB"
                progress_bar.update(train_batch_size)
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=args.revision)

                logs = {"epoch_loss": loss_total / len(train_dataloader)}
                accelerator.log(logs, step=global_step)

                status.job_count = max_train_steps
                status.job_no = global_step
                status.textinfo = f"Steps: {global_step}/{max_train_steps} (Current)," \
                                  f" {args.revision}/{lifetime_step + max_train_steps} (Lifetime), Epoch: {global_epoch}"

                # Log completion message
                if training_complete or status.interrupted:
                    print("  Training complete (step check).")
                    if status.interrupted:
                        state = "cancelled"
                    else:
                        state = "complete"

                    status.textinfo = f"Training {state} {global_step}/{max_train_steps}, {args.revision}" \
                                      f" total."

                    break

            accelerator.wait_for_everyone()

            args.epoch += 1
            global_epoch += 1

            status.job_count = max_train_steps
            status.job_no = global_step

            check_save(progress_bar, True)

            if args.num_train_epochs > 1:
                training_complete = global_epoch >= max_train_epochs

            if training_complete or status.interrupted:
                print("  Training complete (step check).")
                if status.interrupted:
                    state = "cancelled"
                else:
                    state = "complete"

                status.textinfo = f"Training {state} {global_step}/{max_train_steps}, {args.revision}" \
                                  f" total."

                break

            # Do this at the very END of the epoch, only after we're sure we're not done
            if args.epoch_pause_frequency > 0 and args.epoch_pause_time > 0:
                if not global_epoch % args.epoch_pause_frequency:
                    print(f"Giving the GPU a break for {args.epoch_pause_time} seconds.")
                    for i in range(args.epoch_pause_time):
                        if status.interrupted:
                            training_complete = True
                            print("Training complete, interrupted.")
                            break
                        time.sleep(1)


        print(f"Profile memory is {profile_memory}")
        if profile_memory:
            print("Stopping profiler.")
            prof.stop()
            print("No, really, stopped the damned profiler.")
        cleanup_memory()
        accelerator.end_training()
        print("Ended training.")
        result.msg = msg
        result.config = args
        result.samples = last_samples
        return result

    return inner_loop()
