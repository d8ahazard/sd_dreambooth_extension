import gc
import json
import os
from typing import Optional

import torch
import torch.utils.checkpoint
from huggingface_hub import HfFolder, whoami

from dreambooth.db_config import DreamboothConfig
from dreambooth.train_dreambooth import main
from modules import paths, shared, devices

try:
    cmd_dreambooth_models_path = shared.cmd_opts.dreambooth_models_path
except:
    cmd_dreambooth_models_path = None

mem_record = {}


def printm(msg, reset=False):
    global mem_record
    if reset:
        mem_record = {}
    allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
    cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
    mem_record[msg] = f"{allocated}/{cached}GB"
    print(f' {msg} \n Allocated: {allocated}GB \n Reserved: {cached}GB \n')


def dumb_safety(images, clip_input):
    return images, False


def isset(val: str):
    return val is not None and val != "" and val != "*"


def load_params(pretrained_model_name_or_path,
                pretrained_vae_name_or_path,
                instance_data_dir,
                class_data_dir,
                instance_prompt,
                class_prompt,
                save_sample_prompt,
                save_sample_negative_prompt,
                n_save_sample,
                save_guidance_scale,
                save_infer_steps,
                num_class_images,
                resolution,
                center_crop,
                train_text_encoder,
                train_batch_size,
                sample_batch_size,
                num_train_epochs,
                max_train_steps,
                gradient_accumulation_steps,
                gradient_checkpointing,
                learning_rate,
                scale_lr,
                lr_scheduler,
                lr_warmup_steps,
                use_8bit_adam,
                adam_beta1,
                adam_beta2,
                adam_weight_decay,
                adam_epsilon,
                max_grad_norm,
                save_preview_every,
                save_embedding_every,
                mixed_precision,
                not_cache_latents,
                concepts_list,
                use_cpu,
                pad_tokens,
                max_token_length,
                hflip,
                use_ema):
    tc = DreamboothConfig()

    tc.from_ui(pretrained_model_name_or_path,
               pretrained_vae_name_or_path,
               instance_data_dir,
               class_data_dir,
               instance_prompt,
               class_prompt,
               save_sample_prompt,
               save_sample_negative_prompt,
               n_save_sample,
               save_guidance_scale,
               save_infer_steps,
               num_class_images,
               resolution,
               center_crop,
               train_text_encoder,
               train_batch_size,
               sample_batch_size,
               num_train_epochs,
               max_train_steps,
               gradient_accumulation_steps,
               gradient_checkpointing,
               learning_rate,
               scale_lr,
               lr_scheduler,
               lr_warmup_steps,
               use_8bit_adam,
               adam_beta1,
               adam_beta2,
               adam_weight_decay,
               adam_epsilon,
               max_grad_norm,
               save_preview_every,  # Replaces save_interval, save_min_steps
               save_embedding_every,
               mixed_precision,
               not_cache_latents,
               concepts_list,
               use_cpu,
               pad_tokens,
               max_token_length,
               hflip,
               use_ema)

    target_values = ["pretrained_vae_name_or_path",
                     "instance_data_dir",
                     "class_data_dir",
                     "instance_prompt",
                     "class_prompt",
                     "save_sample_prompt",
                     "save_sample_negative_prompt",
                     "n_save_sample",
                     "save_guidance_scale",
                     "save_infer_steps",
                     "num_class_images",
                     "resolution",
                     "center_crop",
                     "train_text_encoder",
                     "train_batch_size",
                     "sample_batch_size",
                     "num_train_epochs",
                     "max_train_steps",
                     "gradient_accumulation_steps",
                     "gradient_checkpointing",
                     "learning_rate",
                     "scale_lr",
                     "lr_scheduler",
                     "lr_warmup_steps",
                     "use_8bit_adam",
                     "adam_beta1",
                     "adam_beta2",
                     "adam_weight_decay",
                     "adam_epsilon",
                     "max_grad_norm",
                     "save_preview_every",
                     "save_embedding_every",
                     "mixed_precision",
                     "not_cache_latents",
                     "concepts_list",
                     "use_cpu",
                     "pad_tokens",
                     "max_token_length",
                     "hflip",
                     "use_ema"]

    data = tc.from_file(pretrained_model_name_or_path)
    values = []
    for target in target_values:
        if target in data:
            values.append(data[target])
        else:
            values.append(None)
    values.append(f"Loaded params from {pretrained_model_name_or_path}.")
    return values


def get_db_models():
    model_dir = os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path
    out_dir = os.path.join(model_dir, "dreambooth")
    output = []
    if os.path.exists(out_dir):
        dirs = os.listdir(out_dir)
        for found in dirs:
            if os.path.isdir(os.path.join(out_dir, found)):
                output.append(found)
    return output


def start_training(pretrained_model_name_or_path,
                   pretrained_vae_name_or_path,
                   instance_data_dir,
                   class_data_dir,
                   instance_prompt,
                   class_prompt,
                   save_sample_prompt,
                   save_sample_negative_prompt,
                   n_save_sample,
                   save_guidance_scale,
                   save_infer_steps,
                   num_class_images,
                   resolution,
                   center_crop,
                   train_text_encoder,
                   train_batch_size,
                   sample_batch_size,
                   num_train_epochs,
                   max_train_steps,
                   gradient_accumulation_steps,
                   gradient_checkpointing,
                   learning_rate,
                   scale_lr,
                   lr_scheduler,
                   lr_warmup_steps,
                   use_8bit_adam,
                   adam_beta1,
                   adam_beta2,
                   adam_weight_decay,
                   adam_epsilon,
                   max_grad_norm,
                   save_preview_every,  # Replaces save_interval, save_min_steps
                   save_embedding_every,
                   mixed_precision,
                   not_cache_latents,
                   concepts_list,
                   use_cpu,
                   pad_tokens,
                   max_token_length,
                   hflip,
                   use_ema
                   ):
    if pretrained_model_name_or_path == "" or pretrained_model_name_or_path is None:
        print("Invalid model name.")
        return "Create or select a model first.", ""
    config = DreamboothConfig().from_file(pretrained_model_name_or_path)

    if config is None:
        print("Unable to load config?")
        return "Invalid source checkpoint", ""

    total_steps = config["total_steps"]
    config.from_ui(pretrained_model_name_or_path,
                   pretrained_vae_name_or_path,
                   instance_data_dir,
                   class_data_dir,
                   instance_prompt,
                   class_prompt,
                   save_sample_prompt,
                   save_sample_negative_prompt,
                   n_save_sample,
                   save_guidance_scale,
                   save_infer_steps,
                   num_class_images,
                   resolution,
                   center_crop,
                   train_text_encoder,
                   train_batch_size,
                   sample_batch_size,
                   num_train_epochs,
                   max_train_steps,
                   gradient_accumulation_steps,
                   gradient_checkpointing,
                   learning_rate,
                   scale_lr,
                   lr_scheduler,
                   lr_warmup_steps,
                   use_8bit_adam,
                   adam_beta1,
                   adam_beta2,
                   adam_weight_decay,
                   adam_epsilon,
                   max_grad_norm,
                   save_preview_every,  # Replaces save_interval, save_min_steps
                   save_embedding_every,
                   mixed_precision,
                   not_cache_latents,
                   concepts_list,
                   use_cpu,
                   pad_tokens,
                   max_token_length,
                   hflip,
                   use_ema)
    config.save()
    msg = None
    if not isset(instance_data_dir) and not isset(concepts_list):
        msg = "No instance data specified."
    if not isset(instance_prompt) and not isset(concepts_list):
        msg = "No instance prompt specified."
    if not os.path.exists(config.working_dir):
        msg = "Invalid training data directory."
    if isset(pretrained_vae_name_or_path) and not os.path.exists(pretrained_vae_name_or_path):
        msg = "Invalid Pretrained VAE Path."
    if resolution <= 0:
        msg = "Invalid resolution."
    if isset(concepts_list):
        concepts_loaded = False
        try:
            alist = str(config.concepts_list)
            if "'" in alist:
                alist = alist.replace("'", '"')
            print(f"Trying to parse: {alist}")
            concepts_list = json.loads(alist)
            concepts_loaded = True
        except:
            pass

        if not concepts_loaded:
            try:
                if os.path.exists(concepts_list):
                    with open(concepts_list, "r") as f:
                        concepts_list = json.load(f)
                    concepts_loaded = True
            except:
                print("Unable to load concepts from file either, this is bad.")
                pass
        if not concepts_loaded:
            msg = "Unable to parse concepts list."

    if msg:
        shared.state.textinfo = msg
        print(msg)
        return msg, ""

    # Clear memory and do "stuff" only after we've ensured all the things are right
    print("Starting Dreambooth training...")
    shared.sd_model.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()
    printm("VRAM cleared.", True)

    shared.state.textinfo = "Initializing dreambooth training..."
    trained_steps = main(config)
    total_steps += trained_steps
    if config["total_steps"] != total_steps:
        config["total_steps"] = total_steps
        config.save()

    devices.torch_gc()
    gc.collect()
    printm("Training completed, reloading SD Model.")
    print(f'Memory output: {mem_record}')
    shared.sd_model.to(shared.device)
    print("Re-applying optimizations...")
    res = f"Training {'interrupted' if shared.state.interrupted else 'finished'}. " \
          f"Total lifetime steps: {total_steps} \n"
    print(f"Returning result: {res}")
    return res, ""


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"
