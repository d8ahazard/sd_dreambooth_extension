import gc
import logging
import math
import os
import random
import traceback
from pathlib import Path
from typing import Optional

import torch
import torch.utils.checkpoint
from PIL import features
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import logging as dl
from huggingface_hub import HfFolder, whoami
from six import StringIO
from transformers import CLIPTextModel

from extensions.sd_dreambooth_extension.dreambooth import conversion
from extensions.sd_dreambooth_extension.dreambooth.db_config import from_file, Concept
from modules import paths, shared, devices, sd_models, generation_parameters_copypaste

try:
    cmd_dreambooth_models_path = shared.cmd_opts.dreambooth_models_path
except:
    cmd_dreambooth_models_path = None

logger = logging.getLogger(__name__)
# define a Handler which writes DEBUG messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.setLevel(logging.DEBUG)
dl.set_verbosity_error()

mem_record = {}


def sanitize_name(name):
    return "".join(x for x in name if (x.isalnum() or x in "._- "))


def log_memory():
    mem = printm("", True)
    return f"Current memory usage: {mem}"


def generate_sample_img(model_dir: str):
    print("Gensample?")
    if model_dir is None or model_dir == "":
        return "Please select a model."
    config = from_file(model_dir)
    unload_system_models()
    model_path = config.pretrained_model_name_or_path
    if not os.path.exists(config.pretrained_model_name_or_path):
        print(f"Model path '{config.pretrained_model_name_or_path}' doesn't exist.")
        return f"Can't find diffusers model at {config.pretrained_model_name_or_path}."
    try:
        print(f"Loading model from {model_path}.")
        text_enc_model = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path,
                                                       subfolder="text_encoder", revision=config.revision)
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            text_encoder=text_enc_model,
            torch_dtype=torch.float16,
            revision=config.revision,
            safety_checker=None,
			feature_extractor=None,
			requires_safety_checker=False
        )
        pipeline = pipeline.to(shared.device)
        pil_features = list_features()
        save_dir = os.path.join(shared.sd_path, "outputs", "dreambooth")
        for concept in config.concepts_list:
            save_sample_prompt = concept.save_sample_prompt
            db_model_path = config.model_dir
            if save_sample_prompt is None:
                msg = "Please provide a sample prompt."
                print(msg)
                return msg
            shared.state.textinfo = f"Generating preview image for model {db_model_path}..."
            seed = concept.sample_seed
            # I feel like this might not actually be necessary...but what the heck.
            if seed is None or seed == '' or seed == -1:
                seed = int(random.randrange(21474836147))
            g_cuda = torch.Generator(device=shared.device).manual_seed(seed)
            sample_dir = os.path.join(save_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            file_count = 0
            for x in Path(sample_dir).iterdir():
                if is_image(x, pil_features):
                    file_count += 1
            shared.state.job_count = concept.n_save_sample
            for n in range(concept.n_save_sample):
                file_count += 1
                shared.state.job_no = n
                image = pipeline(save_sample_prompt, num_inference_steps=concept.save_infer_steps,
                                 guidance_scale=concept.save_guidance_scale,
                                 scheduler=EulerAncestralDiscreteScheduler(beta_start=0.00085,
                                                                           beta_end=0.012),
                                 negative_prompt=concept.save_sample_negative_prompt,
                                 generator=g_cuda).images[0]

                if shared.opts.enable_pnginfo:
                    params = {
                        "Steps": concept.save_infer_steps,
                        "Sampler": "Euler A",
                        "CFG scale": concept.save_guidance_scale,
                        "Seed": concept.sample_seed
                    }
                    generation_params_text = ", ".join(
                        [k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in
                         params.items() if v is not None])

                    negative_prompt_text = "\nNegative prompt: " + concept.save_sample_negative_prompt

                    data = f"{save_sample_prompt}{negative_prompt_text}\n{generation_params_text}".strip()
                    image.info["parameters"] = data

                shared.state.current_image = image
                shared.state.textinfo = save_sample_prompt
                sanitized_prompt = sanitize_name(save_sample_prompt)
                image.save(os.path.join(sample_dir, f"{str(file_count).zfill(3)}-{sanitized_prompt}.png"))
    except:
        print("Exception generating sample!")
        traceback.print_exc()
    reload_system_models()
    return "Sample generated."


def cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except:
        pass
    printm("Cleanup completed.")


def unload_system_models():
    if shared.sd_model is not None:
        shared.sd_model.to("cpu")
    for former in shared.face_restorers:
        try:
            former.send_model_to("cpu")
        except:
            pass
    cleanup()
    printm("", True)


def reload_system_models():
    if shared.sd_model is not None:
        shared.sd_model.to(shared.device)
    printm("Restored system models.")


# Borrowed from https://wandb.ai/psuraj/dreambooth/reports/Training-Stable-Diffusion-with-Dreambooth
# --VmlldzoyNzk0NDc3#tl,dr; and https://www.reddit.com/r/StableDiffusion/comments/ybxv7h/good_dreambooth_formula/
def training_wizard_person(
        model_dir
):
    return training_wizard(
        model_dir,
        is_person=True)


def training_wizard(
        model_dir,
        is_person=False
):
    """
    Calculate the number of steps based on our learning rate, return the following:
    status,
    max_train_steps,
    c1_max_steps,
    c1_num_class_images,
    c2_max_steps,
    c2_num_class_images,
    c3_max_steps,
    c3_num_class_images
    """
    if model_dir == "" or model_dir is None:
        return "Please select a model.", 1000, -1, 0, -1, 0, -1, 0
    # Load config, get total steps
    config = from_file(model_dir)

    # Configure generic outputs
    class_steps = []
    class_concepts = []
    if config is None:
        status = "Unable to load config."
        return status, 1000, -1, 0, -1, 0, -1, 0
    else:
        rev = config.revision
        if rev == '' or rev is None:
            rev = 0
        total_steps = int(rev)
        # Build concepts list using current settings
        concepts = config.concepts_list
        pil_feats = list_features()

        # Count the total number of images in all datasets
        total_images = 0
        image_counts = []
        max_images = 0

        for concept in concepts:
            image_count = 0
            if not os.path.exists(concept["instance_data_dir"]):
                print("Nonexistent instance directory.")
            else:
                for x in Path(concept["instance_data_dir"]).iterdir():
                    if is_image(x, pil_feats):
                        total_images += 1
                        image_count += 1
            if image_count > max_images:
                max_images = image_count
            image_counts.append(image_count)

        if total_images == 0:
            print("No training images found, can't do math.")
            return "No training images found, can't do math.", 1000, -1, 0, -1, 0, -1, 0

        # Set "base" value
        magick_number = 50000000
        required_steps = round(total_images * magick_number * config.learning_rate, -2)
        if is_person:
            num_class_images = round(total_images * 12, -1)
        else:
            num_class_images = 0
            required_steps = round(required_steps * 1.5, -2)

        c_idx = 0

        for _ in concepts:
            num_images = image_counts[c_idx]
            if num_images == max_images:
                c_steps = -1
            else:
                c_steps = round(num_images * magick_number * config.learning_rate, -2)
            if is_person:
                c_class_images = round(num_images * 12, -1)
            else:
                c_class_images = 0
            if c_idx < 3:
                class_steps.append(c_steps)
                class_concepts.append(c_class_images)
            c_idx += 1
        c1_steps = class_steps[0] if len(class_steps) > 0 else -1
        c2_steps = class_steps[1] if len(class_steps) > 1 else -1
        c3_steps = class_steps[2] if len(class_steps) > 2 else -1
        c1_class = class_concepts[0] if len(class_concepts) > 0 else 0
        c2_class = class_concepts[1] if len(class_concepts) > 1 else 0
        c3_class = class_concepts[2] if len(class_concepts) > 2 else 0
        # Ensure we don't over-train?
        if total_steps >= required_steps:
            required_steps = 0
        else:
            required_steps = required_steps - total_steps
        status = f"Wizard completed, using {required_steps} lifetime steps and {num_class_images} class images."
    return status, required_steps, c1_steps, c1_class, c2_steps, c2_class, c3_steps, c3_class


def performance_wizard():
    status = ""
    attention = "flash_attention"
    gradient_checkpointing = True
    mixed_precision = 'fp16'
    not_cache_latents = True
    sample_batch_size = 1
    train_batch_size = 1
    train_text_encoder = False
    use_8bit_adam = True
    use_cpu = False
    use_ema = False
    gb = 0
    try:
        t = torch.cuda.get_device_properties(0).total_memory
        gb = math.ceil(t / 1073741824)
        print(f"Total VRAM: {gb}")
        if gb >= 24:
            attention = "default"
            gradient_checkpointing = False
            mixed_precision = 'no'
            not_cache_latents = False
            sample_batch_size = 4
            train_batch_size = 2
            train_text_encoder = True
            use_ema = True
            use_8bit_adam = False
        if 24 > gb >= 16:
            attention = "xformers"
            not_cache_latents = False
            train_text_encoder = True
            use_ema = True
        if 16 > gb >= 10:
            train_text_encoder = False
            use_ema = False
        if gb < 10:
            use_cpu = True
            use_8bit_adam = False
            mixed_precision = 'no'

    except:
        pass
    msg = f"Calculated training params based on {gb}GB of VRAM detected."

    has_xformers = False
    try:
        if shared.cmd_opts.xformers or shared.cmd_opts.force_enable_xformers:
            import xformers
            import xformers.ops
            has_xformers = shared.cmd_opts.xformers or shared.cmd_opts.force_enable_xformers
    except:
        pass
    if has_xformers:
        use_8bit_adam = True
        mixed_precision = "fp16"
        msg += "<br>Xformers detected, enabling 8Bit Adam and setting mixed precision to 'fp16'"

    if use_cpu:
        msg += "<br>Detected less than 10GB of VRAM, setting CPU training to true."
    return status, attention, gradient_checkpointing, mixed_precision, not_cache_latents, sample_batch_size, \
        train_batch_size, train_text_encoder, use_8bit_adam, use_cpu, use_ema


def printm(msg="", reset=False):
    global mem_record
    try:
        allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
        reserved = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
        if not mem_record:
            mem_record = {}
        if reset:
            max_allocated = round(torch.cuda.max_memory_allocated(0) / 1024 ** 3, 1)
            max_reserved = round(torch.cuda.max_memory_reserved(0) / 1024 ** 3, 1)
            output = f" Allocated {allocated}/{max_allocated}GB \n Reserved: {reserved}/{max_reserved}GB \n"
            torch.cuda.reset_peak_memory_stats()
            print(output)
            mem_record = {}
        else:
            mem_record[msg] = f"{allocated}/{reserved}GB"
            output = f' {msg} \n Allocated: {allocated}GB \n Reserved: {reserved}GB \n'
            print(output)
    except:
        output = "Error parsing memory stats. Do you have a NVIDIA GPU?"
    return output


def dumb_safety(images, clip_input):
    return images, False


def isset(val: str):
    return val is not None and val != "" and val != "*"


def list_features():
    # Create buffer for pilinfo() to write into rather than stdout
    buffer = StringIO()
    features.pilinfo(out=buffer)
    pil_features = []
    # Parse and analyse lines
    for line in buffer.getvalue().splitlines():
        if "Extensions:" in line:
            ext_list = line.split(": ")[1]
            extensions = ext_list.split(", ")
            for extension in extensions:
                if extension not in pil_features:
                    pil_features.append(extension)
    return pil_features


def is_image(path: Path, feats=None):
    if feats is None:
        feats = []
    if not len(feats):
        feats = list_features()
    is_img = path.is_file() and path.suffix.lower() in feats
    return is_img


def load_params(model_dir):
    data = from_file(model_dir)
    msg = ""
    if data is None:
        print("Can't load config!")
        msg = "Please specify a model to load."

    concepts = []
    ui_dict = {}
    for key in data.__dict__:
        value = data.__dict__[key]
        if key == "concepts_list":
            concepts = value
        else:
            if key == "pretrained_model_name_or_path":
                key = "model_path"
            ui_dict[f"db_{key}"] = value
            msg = "Loaded config."

    ui_concepts = concepts if concepts is not None else []
    if len(ui_concepts) < 3:
        while len(ui_concepts) < 3:
            ui_concepts.append(Concept())
    c_idx = 1
    for ui_concept in ui_concepts:
        if c_idx > 3:
            break

        for key in sorted(ui_concept):
            ui_dict[f"c{c_idx}_{key}"] = ui_concept[key]
        c_idx += 1
    ui_dict["db_status"] = msg
    ui_keys = ["db_adam_beta1", "db_adam_beta2", "db_adam_epsilon", "db_adam_weight_decay", "db_attention",
               "db_center_crop", "db_concepts_path", "db_gradient_accumulation_steps", "db_gradient_checkpointing",
               "db_half_model", "db_has_ema", "db_hflip", "db_learning_rate", "db_lr_scheduler", "db_lr_warmup_steps",
               "db_max_grad_norm", "db_max_token_length", "db_max_train_steps", "db_mixed_precision", "db_model_path",
               "db_not_cache_latents", "db_num_train_epochs", "db_pad_tokens", "db_pretrained_vae_name_or_path",
               "db_prior_loss_weight", "db_resolution", "db_revision", "db_sample_batch_size",
               "db_save_embedding_every", "db_save_preview_every", "db_scale_lr", "db_scheduler", "db_src",
               "db_train_batch_size", "db_train_text_encoder", "db_use_8bit_adam", "db_use_concepts", "db_use_cpu",
               "db_use_ema", "db_v2", "c1_class_data_dir", "c1_class_guidance_scale", "c1_class_infer_steps",
               "c1_class_negative_prompt", "c1_class_prompt", "c1_class_token", "c1_file_prompt_contents",
               "c1_instance_data_dir", "c1_instance_prompt", "c1_instance_token", "c1_max_steps", "c1_n_save_sample",
               "c1_num_class_images", "c1_sample_seed", "c1_save_guidance_scale", "c1_save_infer_steps",
               "c1_save_sample_negative_prompt", "c1_save_sample_prompt", "c1_save_sample_template", "c2_class_data_dir",
               "c2_class_guidance_scale", "c2_class_infer_steps", "c2_class_negative_prompt", "c2_class_prompt",
               "c2_class_token", "c2_file_prompt_contents", "c2_instance_data_dir", "c2_instance_prompt",
               "c2_instance_token", "c2_max_steps", "c2_n_save_sample", "c2_num_class_images", "c2_sample_seed",
               "c2_save_guidance_scale", "c2_save_infer_steps", "c2_save_sample_negative_prompt",
               "c2_save_sample_prompt", "c2_save_sample_template", "c3_class_data_dir", "c3_class_guidance_scale",
               "c3_class_infer_steps", "c3_class_negative_prompt", "c3_class_prompt", "c3_class_token",
               "c3_file_prompt_contents", "c3_instance_data_dir", "c3_instance_prompt", "c3_instance_token",
               "c3_max_steps", "c3_n_save_sample", "c3_num_class_images", "c3_sample_seed", "c3_save_guidance_scale",
               "c3_save_infer_steps", "c3_save_sample_negative_prompt", "c3_save_sample_prompt",
               "c3_save_sample_template", "db_status"]
    output = []
    for key in ui_keys:
        if key in ui_dict:
            output.append(ui_dict[key])
        else:
            output.append(None)
    print(f"Returning {output}")
    return output


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


def start_training(model_dir: str, imagic_only: bool):
    global mem_record
    if model_dir == "" or model_dir is None:
        print("Invalid model name.")
        return "Create or select a model first.", ""
    config = from_file(model_dir)

    # Clear pretrained VAE Name if applicable
    if config.pretrained_vae_name_or_path == "":
        config.pretrained_vae_name_or_path = None

    msg = None
    if config.attention == "xformers":
        if config.mixed_precision == "no":
            msg = "Using xformers, please set mixed precision to 'fp16' to continue."
        if not shared.cmd_opts.xformers and not shared.cmd_opts.force_enable_xformers:
            msg = "Xformers is not enabled, please relaunch using the --xformers command-line argument to continue."
    if config.use_cpu:
        if config.use_8bit_adam or config.mixed_precision != "no":
            msg = "CPU Training detected, please disable 8Bit Adam and set mixed precision to 'no' to continue."
    if not len(config.concepts_list):
        msg = "Please configure some concepts."
    if not os.path.exists(config.pretrained_model_name_or_path):
        msg = "Invalid training data directory."
    if isset(config.pretrained_vae_name_or_path) and not os.path.exists(config.pretrained_vae_name_or_path):
        msg = "Invalid Pretrained VAE Path."
    if config.resolution <= 0:
        msg = "Invalid resolution."

    if msg:
        shared.state.textinfo = msg
        print(msg)
        return msg, msg, 0, ""

    # Clear memory and do "stuff" only after we've ensured all the things are right
    print("Starting Dreambooth training...")
    unload_system_models()
    total_steps = config.revision
    if imagic_only:
        shared.state.textinfo = "Initializing imagic training..."
        print(shared.state.textinfo)
        from extensions.sd_dreambooth_extension.dreambooth.train_imagic import train_imagic
        mem_record = train_imagic(config, mem_record)
    else:
        shared.state.textinfo = "Initializing dreambooth training..."
        print(shared.state.textinfo)
        from extensions.sd_dreambooth_extension.dreambooth.train_dreambooth import main
        config, mem_record, msg = main(config, mem_record)
        if config.revision != total_steps:
            config.save()
    total_steps = config.revision
    devices.torch_gc()
    gc.collect()
    printm("Training completed, reloading SD Model.")
    print(f'Memory output: {mem_record}')
    reload_system_models()
    res = f"Training {'interrupted' if shared.state.interrupted else 'finished'}. " \
          f"Total lifetime steps: {total_steps} \n"
    print(f"Returning result: {res}")
    return res, "", total_steps, ""


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def save_checkpoint(model_name: str, total_steps: int):
    print(f"Successfully trained model for a total of {total_steps} steps, converting to ckpt.")
    ckpt_dir = shared.cmd_opts.ckpt_dir
    models_path = os.path.join(paths.models_path, "Stable-diffusion")
    if ckpt_dir is not None:
        models_path = ckpt_dir
    src_path = os.path.join(
        os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path, "dreambooth",
        model_name, "working")
    out_file = os.path.join(models_path, f"{model_name}_{total_steps}.ckpt")
    conversion.compile_checkpoint(model_name)
    sd_models.list_models()
