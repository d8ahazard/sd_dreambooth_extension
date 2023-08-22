import glob
import importlib
import importlib.util
import json
import logging
import math
import os
import random
import sys
import traceback
from collections import OrderedDict

import torch
import torch.utils.data.dataloader
from accelerate import find_executable_batch_size
from diffusers.utils import logging as dl
from torch.optim import AdamW

from dreambooth import shared
from dreambooth.dataclasses import db_config
from dreambooth.dataclasses.db_config import from_file, sanitize_name
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.dataset.bucket_sampler import BucketSampler
from dreambooth.dataset.class_dataset import ClassDataset
from dreambooth.optimization import UniversalScheduler
from dreambooth.sd_to_diff import extract_checkpoint
from dreambooth.shared import status, run
from dreambooth.utils.gen_utils import generate_dataset, generate_classifiers
from dreambooth.utils.image_utils import (
    get_images,
    db_save_image,
    make_bucket_resolutions,
    get_dim,
    closest_resolution,
    open_and_trim,
)
from dreambooth.utils.model_utils import (
    unload_system_models,
    reload_system_models,
    get_lora_models,
    get_checkpoint_match,
    get_model_snapshots,
    LORA_SHARED_SRC_CREATE, get_db_models,
)
from dreambooth.utils.utils import printm, cleanup
from helpers.image_builder import ImageBuilder
from helpers.mytqdm import mytqdm

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.setLevel(logging.DEBUG)
dl.set_verbosity_error()


def gr_update(default=None, **kwargs):
    try:
        import gradio

        return gradio.update(visible=True, **kwargs)
    except:
        return kwargs["value"] if "value" in kwargs else default


def training_wizard_person(model_dir):
    return training_wizard(model_dir, is_person=True)


def training_wizard(model_dir, is_person=False):
    """
    Calculate the number of steps based on our learning rate, return the following:
    db_num_train_epochs,
    c1_num_class_images_per,
    c2_num_class_images_per,
    c3_num_class_images_per,
    c4_num_class_images_per,
    db_status
    """
    if model_dir == "" or model_dir is None:
        return 100, 0, 0, 0, 0, "Please select a model."
    step_mult = 150 if is_person else 100

    if is_person:
        class_count = 5
    else:
        class_count = 0

    w_status = f"Wizard results:"
    w_status += f"<br>Num Epochs: {step_mult}"
    w_status += f"<br>Num instance images per class image: {class_count}"

    print(w_status)

    return int(step_mult), class_count, class_count, class_count, class_count, w_status


def largest_prime_factor(n):
    # Special case for n = 2
    if n == 2:
        return 2

    # Start with the first prime number, 2
    largest_factor = 2

    # Divide n by 2 as many times as possible
    while n % 2 == 0:
        n = n // 2

    # Check the remaining odd factors of n
    for i in range(3, int(n ** 0.5) + 1, 2):
        # Divide n by i as many times as possible
        while n % i == 0:
            largest_factor = i
            n = n // i

    # If there is a prime factor larger than the square root, it will be the remaining value of n
    if n > 2:
        largest_factor = n

    return largest_factor


def closest_factors_to_sqrt(n):
    # Find the square root of n
    sqrt_n = int(n ** 0.5)

    # Initialize the factors to the square root and 1
    f1, f2 = sqrt_n, 1

    # Check if n is a prime number
    if math.sqrt(n) == sqrt_n:
        return sqrt_n, sqrt_n

    # Find the first pair of factors that are closest in value
    while n % f1 != 0:
        f1 -= 1
        f2 = n // f1

    # Initialize the closest difference to the difference between the square root and f1
    closest_diff = abs(sqrt_n - f1)
    closest_factors = (f1, f2)

    # Check the pairs of factors below the square root
    for i in range(sqrt_n - 1, 1, -1):
        if n % i == 0:
            # Calculate the difference between the square root and the factors
            diff = min(abs(sqrt_n - i), abs(sqrt_n - (n // i)))
            # Update the closest difference and factors if necessary
            if diff < closest_diff:
                closest_diff = diff
                closest_factors = (i, n // i)

    return closest_factors


def performance_wizard(model_name):
    """
    Calculate performance settings based on available resources.
    @return:
    attention: Memory Attention
    optimizer: Optimizer
    gradient_checkpointing: Whether to use gradient checkpointing or not.
    gradient_accumulation_steps: Number of steps to use. Set to batch size.
    mixed_precision: Mixed precision to use. BF16 will be selected if available.
    not_cache_latents: Latent caching.
    sample_batch_size: Batch size to use when creating class images.
    train_batch_size: Batch size to use when training.
    stop_text_encoder: Whether to train text encoder or not.
    use_lora: Train using LORA. Better than "use CPU".
    use_ema: Train using EMA.
    msg: Stuff to show in the UI
    """
    attention = "flash_attention"
    optimizer = "8bit AdamW"
    gradient_checkpointing = False
    gradient_accumulation_steps = 1
    mixed_precision = "fp16"
    cache_latents = True
    sample_batch_size = 1
    train_batch_size = 1
    stop_text_encoder = 0
    use_lora = False
    use_ema = False
    config = None
    if model_name == "" or model_name is None:
        print("Can't load config, specify a model name!")
    else:
        config = from_file(model_name)
    save_samples_every = gr_update(config.save_preview_every)
    save_weights_every = gr_update(config.save_embedding_every)

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        mixed_precision = "bf16"
    if config is not None:
        total_images = 0
        for concept in config.concepts():
            idd = concept.instance_data_dir
            if idd and os.path.exists(idd):
                images = get_images(idd)
                total_images += len(images)
        print(f"Total images: {total_images}")
        if total_images != 0:
            best_factors = closest_factors_to_sqrt(total_images)
            largest_prime = largest_prime_factor(total_images)
            largest_factor = (
                best_factors[0]
                if best_factors[0] > best_factors[1]
                else best_factors[1]
            )
            smallest_factor = (
                best_factors[0]
                if best_factors[0] < best_factors[1]
                else best_factors[1]
            )
            factor_diff = largest_factor - smallest_factor
            print(f"Largest prime: {largest_prime}")
            print(f"Best factors: {best_factors}")
            if largest_prime <= factor_diff:
                train_batch_size = largest_prime
                gradient_accumulation_steps = largest_prime
            else:
                train_batch_size = largest_factor
                gradient_accumulation_steps = smallest_factor

    has_xformers = False
    try:
        from diffusers.utils.import_utils import is_xformers_available

        has_xformers = is_xformers_available()
    except:
        pass
    if has_xformers:
        attention = "xformers"
    try:
        stop_text_encoder = 0.75
        t = torch.cuda.get_device_properties(0).total_memory
        gb = math.ceil(t / 1073741824)
        print(f"Total VRAM: {gb}")
        if gb >= 24:
            sample_batch_size = 4
            use_ema = True
            if attention != "xformers":
                attention = "no"
                train_batch_size = 1
                gradient_accumulation_steps = 1
        if 24 > gb >= 16:
            use_ema = True
        if 16 > gb >= 12:
            use_ema = False
            cache_latents = False
            gradient_accumulation_steps = 1
            train_batch_size = 1
        if gb < 12:
            use_lora = True
            save_samples_every = gr_update(value=0)
            save_weights_every = gr_update(value=0)

        msg = f"Calculated training params based on {gb}GB of VRAM:"
    except Exception as e:
        msg = f"An exception occurred calculating performance values: {e}"
        pass

    log_dict = {
        "Attention": attention,
        "Gradient Checkpointing": gradient_checkpointing,
        "Accumulation Steps": gradient_accumulation_steps,
        "Precision": mixed_precision,
        "Cache Latents": cache_latents,
        "Training Batch Size": train_batch_size,
        "Class Generation Batch Size": sample_batch_size,
        "Text Encoder Ratio": stop_text_encoder,
        "Optimizer": optimizer,
        "EMA": use_ema,
        "LORA": use_lora,
    }
    for key in log_dict:
        msg += f"<br>{key}: {log_dict[key]}"
    return (
        attention,
        gradient_checkpointing,
        gradient_accumulation_steps,
        mixed_precision,
        cache_latents,
        optimizer,
        sample_batch_size,
        train_batch_size,
        stop_text_encoder,
        use_lora,
        use_ema,
        save_samples_every,
        save_weights_every,
        msg,
    )


# p,
# overrideDenoising,
# overrideMaskBlur,
# path,
# searchSubdir,
# divider,
# howSplit,
# saveMask,
# pathToSave,
# viewResults,
# saveNoFace,
# onlyMask,
# invertMask,
# singleMaskPerImage,
# countFaces,
# maskSize,
# keepOriginalName,
# pathExisting,
# pathMasksExisting,
# pathToSaveExisting,
# selectedTab,
# faceDetectMode,
# face_x_scale,
# face_y_scale,
# minFace,
# multiScale,
# multiScale2,
# multiScale3,
# minNeighbors,
# mpconfidence,
# mpcount,
# debugSave,
# optimizeDetect


def get_swap_parameters():
    return OrderedDict(
        [
            ("overrideDenoising", True),
            ("overrideMaskBlur", True),
            ("path", "./"),
            ("searchSubdir", False),
            ("divider", 1),
            ("howSplit", "Both ▦"),
            ("saveMask", False),
            ("pathToSave", "./"),
            ("viewResults", False),
            ("saveNoFace", False),
            ("onlyMask", False),
            ("invertMask", False),
            ("singleMaskPerImage", False),
            ("countFaces", False),
            ("maskSize", 0),
            ("keepOriginalName", False),
            ("pathExisting", ""),
            ("pathMasksExisting", ""),
            ("pathToSaveExisting", ""),
            ("selectedTab", "generateMasksTab"),
            ("faceDetectMode", "Normal (OpenCV + FaceMesh)"),
            ("face_x_scale", 1.0),
            ("face_y_scale", 1.0),
            ("minFace", 50),
            ("multiScale", 1.03),
            ("multiScale2", 1.0),
            ("multiScale3", 1.0),
            ("minNeighbors", 5),
            ("mpconfidence", 0.5),
            ("mpcount", 1),
            ("debugSave", False),
            ("optimizeDetect", True),
        ]
    )


def get_script_class():
    script_class = None
    try:
        from modules.scripts import list_scripts

        scripts = list_scripts("scripts", ".py")
        for script_file in scripts:
            if script_file.filename == "batch_face_swap.py":
                path = script_file.path
                module_name = "batch_face_swap"
                spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                script_class = module.Script()
                break
    except Exception as f:
        print(f"Can't check face swap: {f}")
    return script_class


def generate_samples(
        model_name: str,
        prompt: str,
        prompt_file: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_samples: int,
        batch_size: int,
        seed: int,
        steps: int,
        scale: float,
        class_gen_method: str = "Native Diffusers",
        scheduler: str = "UniPCMultistep",
        swap_faces: bool = False,
        swap_prompt: str = "",
        swap_negative: str = "",
        swap_steps: int = 40,
        swap_batch: int = 1,
):
    if batch_size > num_samples:
        batch_size = num_samples

    @find_executable_batch_size(starting_batch_size=batch_size)
    def sample_loop(train_batch_size):
        if model_name is None or model_name == "":
            return "Please select a model."
        config = from_file(model_name)
        source_model = None

        if class_gen_method == "A1111 txt2img (Euler a)":
            tgt_name = (
                model_name if not config.custom_model_name else config.custom_model_name
            )
            tgt_ext = ".safetensors" if config.save_safetensors else ".ckpt"
            if config.use_subdir:
                tgt_file = os.path.join(
                    tgt_name, f"{tgt_name}_{config.revision}{tgt_ext}"
                )
                tgt_file_ema = os.path.join(
                    tgt_name, f"{tgt_name}_{config.revision}_ema{tgt_ext}"
                )
                tgt_file_lora = os.path.join(
                    tgt_name, f"{tgt_name}_{config.revision}_lora{tgt_ext}"
                )
            else:
                tgt_file = f"{tgt_name}{config.revision}{tgt_ext}"
                tgt_file_ema = f"{tgt_name}{config.revision}_ema{tgt_ext}"
                tgt_file_lora = f"{tgt_name}{config.revision}_lora{tgt_ext}"
            for tgt in [tgt_file, tgt_file_ema, tgt_file_lora]:
                model_file = os.path.join(shared.models_path, "Stable-diffusion", tgt)
                print(f"Looking for: {model_file}")
                if os.path.exists(model_file):
                    source_model = model_file

        images = []
        prompts_out = []
        unload_system_models()
        if prompt == "" and prompt_file == "":
            msg = "Please provide a sample prompt or prompt file."
            print(msg)
            return None, None, msg

        if prompt_file == "":
            if ";" in prompt:
                prompts = prompt.split(";")
            else:
                prompts = [prompt]
        else:
            if not os.path.exists(prompt_file):
                msg = "Invalid prompt file."
                print(msg)
                return None, None, msg
            with open(prompt_file, "r") as prompt_data:
                prompts = prompt_data.readlines()
                for i in range(len(prompts)):
                    file_prompt = prompts[i]
                    prompts[i] = file_prompt.replace("[filewords]", prompt).replace(
                        "[name]", prompt
                    )

        try:
            status.textinfo = "Loading diffusion model..."

            img_builder = ImageBuilder(
                config=config,
                class_gen_method=class_gen_method,
                lora_model=config.lora_model_name,
                batch_size=batch_size,
                lora_unet_rank=config.lora_unet_rank,
                lora_txt_rank=config.lora_txt_rank,
                source_checkpoint=source_model,
                scheduler=scheduler,
            )

            prompt_data = []
            for i in range(num_samples):
                sample_prompt = random.choice(prompts)
                pd = PromptData(
                    prompt=sample_prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    scale=scale,
                    out_dir=os.path.join(config.model_dir, "samples"),
                    seed=seed,
                    resolution=(width, height),
                )
                prompt_data.append(pd)

            status.textinfo = (
                f"Generating sample image for model {config.model_name}..."
            )

            pbar = mytqdm(range(num_samples), desc="Generating samples", position=0)
            pbar.reset(num_samples * steps)
            sample_index = 0
            while len(images) < num_samples and not shared.status.interrupted:
                samples_needed = num_samples - len(images)
                to_gen = min(samples_needed, train_batch_size)
                to_generate = []
                batch_images = []
                batch_prompts = []
                print(f"Looping: {len(images)} {to_gen}")

                for i in range(to_gen):
                    sel = prompt_data[sample_index]
                    to_generate.append(sel)
                    batch_prompts.append(sel.prompt)
                    sample_index += 1
                out_images = img_builder.generate_images(to_generate, pbar)
                for img, pd in zip(out_images, to_generate):
                    image_name = db_save_image(img, pd)
                    batch_images.append(image_name)
                images.extend(batch_images)
                prompts_out.extend(batch_prompts)
                shared.status.current_image = images
                shared.status.sample_prompts = batch_prompts
            img_builder.unload(True)
            reload_system_models()
        except Exception as e:
            msg = f"Exception generating sample(s): {e}"
            if "out of memory" in msg:
                print("OOM detected, decreasing batch size.")
                raise
            else:
                print(msg)
                traceback.print_exc()

        try:
            swap_class = get_script_class()
            # if swap_faces and swap_class is not None:
            #     # Get the parent directory of the first image in the list
            #     parent_dir = os.path.dirname(images[0])
            #
            #     # Create the subdirectory called "temp" in the parent directory
            #     temp_dir = os.path.join(parent_dir, "temp")
            #     temp_out = os.path.join(parent_dir, "temp_out")
            #     if not os.path.exists(temp_dir):
            #         os.mkdir(temp_dir)
            #
            #     # Move all images to the "temp" subdirectory
            #     for img_path in images:
            #         img_name = os.path.basename(img_path)
            #         new_path = os.path.join(temp_dir, img_name)
            #         os.rename(img_path, new_path)
            #
            #     # Save the full path of the parent directory + temp as a string to a variable called temp_dir
            #     temp_dir = os.path.abspath(temp_dir)
            #     temp_out = os.path.abspath(temp_out)
            #
            #     from modules.processing import StableDiffusionProcessingImg2Img
            #     p = StableDiffusionProcessingImg2Img(prompt=swap_prompt, negative_prompt=swap_negative, steps=swap_steps, batch_size=swap_batch, sampler_name="Euler a", width=width, height=height, inpaint_full_res=1, inpainting_fill=1)
            #     # p, overrideDenoising, overrideMaskBlur, path, searchSubdir, divider, howSplit, saveMask, pathToSave, viewResults, saveNoFace, onlyMask, invertMask, singleMaskPerImage, countFaces, maskSize, keepOriginalName, pathExisting, pathMasksExisting, pathToSaveExisting, selectedTab, faceDetectMode, face_x_scale, face_y_scale, minFace, multiScale, multiScale2, multiScale3, minNeighbors, mpconfidence, mpcount, debugSave, optimizeDetect
            #     params = get_swap_parameters()
            #     params["path"] = temp_dir
            #     params["pathToSave"] = temp_out
            #
            #     param_list = list(params.values())
            #
            #     foo = swap_class.run(p, *param_list)
            #     print("DO FACE SWAP HERE")
        except Exception as p:
            print(f"Exception face swapping: {p}")
            traceback.print_exc()
            pass

        reload_system_models()
        msg = f"Generated {len(images)} samples."
        print()
        return images, prompts_out, msg

    return sample_loop()


def load_params(model_dir):
    data = from_file(model_dir)
    ui_dict = {}
    msg = ""
    if data is None:
        print("Can't load config!")
        msg = "Please specify a model to load."
    elif data.__dict__ is None:
        print("Can't load config!")
        msg = "Please check your model config."
    else:
        for key in data.__dict__:
            value = data.__dict__[key]
            if key == "pretrained_model_name_or_path":
                key = "model_path"
            ui_dict[f"db_{key}"] = value
            msg = "Loaded config."

    ui_concept_list = data.concepts(4)
    c_idx = 1
    for ui_concept in ui_concept_list:
        for key in sorted(ui_concept.__dict__):
            ui_dict[f"c{c_idx}_{key}"] = ui_concept.__dict__[key]
        c_idx += 1
    ui_dict["db_status"] = msg
    ui_keys = db_config.ui_keys
    output = []
    for key in ui_keys:
        output.append(ui_dict[key] if key in ui_dict else None)

    return output


def load_model_params(model_name):
    """
    @param model_name: The name of the model to load.
    @return:
    db_model_dir: The model directory
    db_model_path: The full path to the model directory
    db_revision: The current revision of the model
    db_v2: If the model requires a v2 config/compilation
    db_has_ema: Was the model extracted with EMA weights
    db_src: The source checkpoint that weights were extracted from or hub URL
    db_shared_diffusers_path:
    db_scheduler: Scheduler used for this model
    db_model_snapshots: A gradio dropdown containing the available snapshots for the model
    db_outcome: The result of loading model params
    """
    if isinstance(model_name, list) and len(model_name) > 0:
        model_name = model_name[0]

    config = from_file(model_name)
    db_model_snapshots = gr_update(choices=[], value="")
    if config is None:
        print("Can't load config!")
        msg = f"Error loading model params: '{model_name}'."
        return "", "", "", "", "", db_model_snapshots, msg
    else:
        snaps = get_model_snapshots(config)
        snap_selection = config.revision if str(config.revision) in snaps else ""
        db_model_snapshots = gr_update(choices=snaps, value=snap_selection)

        loras = get_lora_models(config)
        db_lora_models = gr_update(choices=loras)
        msg = f"Selected model: '{model_name}'."
        return (
            config.model_dir,
            config.revision,
            config.epoch,
            config.model_type,
            "True" if config.has_ema and not config.use_lora else "False",
            config.src,
            config.shared_diffusers_path,
            db_model_snapshots,
            db_lora_models,
            msg,
        )


def start_training(model_dir: str, class_gen_method: str = "Native Diffusers"):
    """

    @param model_dir: The directory containing the dreambooth model/config
    @param class_gen_method: Image Generation Library.
    @return:
    lora_model_name: If using lora, this will be the model name of the saved weights. (For resuming further training)
    revision: The model revision after training.
    epoch: The model epoch after training.
    images: Output images from training.
    status: Any relevant messages.
    """
    if model_dir == "" or model_dir is None:
        print("Invalid model name.")
        msg = "Create or select a model first."
        lora_model_name = gr_update(visible=True)
        return lora_model_name, 0, 0, [], msg
    config = from_file(model_dir)
    # Clear pretrained VAE Name if applicable
    if config.pretrained_vae_name_or_path == "":
        config.pretrained_vae_name_or_path = None

    msg = None
    if config.attention == "xformers":
        if config.mixed_precision == "no":
            msg = "Using xformers, please set mixed precision to 'fp16' or 'bf16' to continue."
    if not len(config.concepts()):
        msg = "Please check your dataset directories."
    if not os.path.exists(config.get_pretrained_model_name_or_path()):
        msg = "Invalid training data directory."
    if config.pretrained_vae_name_or_path:
        if not os.path.exists(config.pretrained_vae_name_or_path):
            msg = "Invalid Pretrained VAE Path."
    if config.resolution <= 0:
        msg = "Invalid resolution."

    if msg:
        print(msg)
        lora_model_name = gr_update(visible=True)
        return lora_model_name, 0, 0, [], msg
    status.begin()
    # Clear memory and do "stuff" only after we've ensured all the things are right
    if config.custom_model_name:
        print(f"Custom model name is {config.custom_model_name}")
    unload_system_models()
    total_steps = config.revision
    config.save(True)
    images = []
    try:
        if config.train_imagic:
            status.textinfo = "Initializing imagic training..."
            print(status.textinfo)
            try:
                from dreambooth.train_imagic import train_imagic  # noqa
            except:
                from dreambooth.train_imagic import train_imagic  # noqa

            result = train_imagic(config)
        else:
            status.textinfo = "Initializing dreambooth training..."
            print(status.textinfo)
            try:
                from dreambooth.train_dreambooth import main  # noqa
            except:
                from dreambooth.train_dreambooth import main  # noqa
            result = main(class_gen_method=class_gen_method)

        config = result.config
        images = result.samples
        if config.revision != total_steps:
            config.save()
        else:
            log_dir = os.path.join(config.model_dir, "logging", "dreambooth", "*")
            list_of_files = glob.glob(log_dir)
            if len(list_of_files):
                latest_file = max(list_of_files, key=os.path.getmtime)
                print(f"No training was completed, deleting log: {latest_file}")
                os.remove(latest_file)
        total_steps = config.revision
        res = (
            f"Training {'interrupted' if status.interrupted else 'finished'}. "
            f"Total lifetime steps: {total_steps} \n"
        )
    except Exception as e:
        res = f"Exception training model: '{e}'."
        traceback.print_exc()
        pass

    status.end()
    cleanup()
    reload_system_models()
    lora_model_name = ""
    if config.lora_model_name:
        lora_model_name = f"{config.model_name}_{total_steps}.pt"
    dirs = get_lora_models()
    lora_model_name = gr_update(choices=sorted(dirs), value=lora_model_name)
    return lora_model_name, total_steps, config.epoch, images, res


def reload_extension():
    ext_name = "extensions.sd_dreambooth_extension"
    deleted = []
    for module in list(sys.modules):
        if module.startswith(ext_name):
            del sys.modules[module]
            deleted.append(module)

    for re_add in deleted:
        try:
            print(f"Replacing: {re_add}")
            importlib.import_module(re_add)

        except Exception as e:
            print(f"Couldn't import module: {re_add}")
    try:
        from postinstall import actual_install  # noqa
    except:
        from dreambooth.postinstall import actual_install  # noqa

    actual_install()


def update_extension():
    git = os.environ.get("GIT", "git")
    ext_dir = os.path.join(shared.script_path, "extensions", "sd_dreambooth_extension")
    run(
        f'"{git}" -C "{ext_dir}" fetch',
        f"Fetching updates...",
        f"Couldn't fetch updates...",
    )
    run(
        f'"{git}" -C "{ext_dir}" pull',
        f"Pulling updates...",
        f"Couldn't pull updates...",
    )
    reload_extension()


def ui_classifiers(model_name: str, class_gen_method: str = "Native Diffusers"):
    """
    UI method for generating class images.
    @param model_name: The model to generate classes for.
    @param class_gen_method" Image Generation Library.
    @return:
    """
    if model_name == "" or model_name is None:
        print("Invalid model name.")
        msg = "Create or select a model first."
        return msg
    config = from_file(model_name)
    status.textinfo = "Generating class images..."
    # Clear pretrained VAE Name if applicable
    if config.pretrained_vae_name_or_path == "":
        config.pretrained_vae_name_or_path = None

    msg = None
    if config.attention == "xformers":
        if config.mixed_precision == "no":
            msg = "Using xformers, please set mixed precision to 'fp16' or 'bf16' to continue."
    if not len(config.concepts()):
        msg = "Please check your dataset directories."
    if not os.path.exists(config.pretrained_model_name_or_path):
        msg = "Invalid training data directory."
    if config.pretrained_vae_name_or_path:
        if not os.path.exists(config.pretrained_vae_name_or_path):
            msg = "Invalid Pretrained VAE Path."
    if config.resolution <= 0:
        msg = "Invalid resolution."

    if msg:
        status.textinfo = msg
        print(msg)
        return [], msg

    images = []
    try:
        unload_system_models()
        count, images = generate_classifiers(config, ui=True)
        reload_system_models()
        msg = f"Generated {count} class images."
    except Exception as e:
        msg = f"Exception generating concepts: {str(e)}"
        traceback.print_exc()
        status.job_no = status.job_count
        status.textinfo = msg
    return images, msg


def start_crop(
        src_dir: str, dest_dir: str, max_res: int, bucket_step: int, dry_run: bool
):
    src_images = get_images(src_dir)

    bucket_resos = make_bucket_resolutions(max_res, bucket_step)

    max_dim = 0
    for (w, h) in bucket_resos:
        if w > max_dim:
            max_dim = w
        if h > max_dim:
            max_dim = h
    _, dirr = os.path.split(src_dir)
    shared.status.begin()
    pbar = mytqdm(src_images, desc=f"Sorting images in directory: {dirr}", position=0)

    out_counts = {}
    out_paths = {}
    for img in src_images:
        pbar.update()
        # Get prompt
        w, h = get_dim(img, max_dim)
        reso = closest_resolution(w, h, bucket_resos)
        if reso in out_counts:
            out_paths[reso].append(img)
        else:
            out_paths[reso] = [img]
        out_counts[reso] = len(out_paths[reso])

    def sort_key(res):
        # Sort by square resolutions first
        if res[0] == res[1]:
            return 0, -res[0]
        # Sort landscape resolutions by height
        elif res[0] > res[1]:
            return 1, -res[1]
        # Sort portrait resolutions by width
        else:
            return 2, -res[0]

    sorted_counts = sorted(
        out_counts.items(), key=lambda x: sort_key(x[0]), reverse=True
    )
    total_images = 0
    for res, count in sorted_counts:
        total_images += count
        print(
            f"RES: {res}  -  {count}  - {[os.path.basename(file) for file in out_paths[res]]}"
        )

    out_images = []
    out_status = ""

    if not dry_run:
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
    pbar.set_description("Cropping images.")

    pbar.reset(total_images)
    for res, images in out_paths.items():
        for image in images:
            pbar.update()
            out_img = os.path.join(
                dest_dir, os.path.splitext(os.path.basename(image))[0] + ".png"
            )
            cropped = open_and_trim(image, res, True)
            if not dry_run:
                print(f"\nSaving to {out_img}")
                cropped.save(out_img, quality=100)
            out_images.append(cropped)
            shared.status.current_image = [cropped]

        out_status = (
            f"{'Saved' if not dry_run else 'Previewed'} {total_images} cropped images."
        )
    status.end()
    return out_status, out_images


def create_model(
        new_model_name: str,
        ckpt_path: str,
        shared_src: str,
        from_hub=False,
        new_model_url="",
        new_model_token="",
        extract_ema=False,
        train_unfrozen=False,
        model_type="v1"
):
    printm("Extracting model.")
    res = 512
    is_512 = model_type == "v1"
    if model_type == "v1x" or model_type=="v2x-512":
        res = 512
    elif model_type == "v2x":
        res = 768
    elif model_type == "sdxl":
        res = 1024
    sh = None
    try:
        from core.handlers.status import StatusHandler
        sh = StatusHandler()
        sh.start(5, "Extracting model")
    except:
        pass
    status.begin()
    if new_model_name is None or new_model_name == "":
        print("No model name.")
        err_msg = "Please select a model"
        if sh is not None:
            sh.end(desc=err_msg)

        return "", "", "", 0, 0, "", "", "", "", res, "", err_msg

    new_model_name = sanitize_name(new_model_name)

    if not from_hub and (shared_src == "" or shared_src == LORA_SHARED_SRC_CREATE):
        checkpoint_info = get_checkpoint_match(ckpt_path)
        if checkpoint_info is None or not os.path.exists(checkpoint_info.filename):
            err_msg = "Unable to find checkpoint file!"
            print(err_msg)
            if sh is not None:
                sh.end(desc=err_msg)
            return "", "", "", 0, 0, "", "", "", "", res, "", err_msg
        ckpt_path = checkpoint_info.filename

    unload_system_models()
    result = extract_checkpoint(new_model_name=new_model_name,
                                checkpoint_file=ckpt_path,
                                extract_ema=extract_ema,
                                train_unfrozen=train_unfrozen,
                                image_size=res,
                                model_type=model_type)
    if result is None:
        err_msg = "Unable to extract checkpoint!"
        print(err_msg)
        if sh is not None:
            sh.end(desc=err_msg)
        return "", "", "", 0, 0, "", "", "", "", res, "", err_msg
    try:
        from core.handlers.models import ModelHandler
        mh = ModelHandler()
        mh.refresh("dreambooth")
    except:
        pass

    cleanup()
    reload_system_models()
    printm("Extraction complete.")
    if sh is not None:
        sh.end(desc="Extraction complete.")
    return gr_update(choices=sorted(get_db_models()), value=new_model_name), \
        result.model_dir, \
        result.revision, \
        result.epoch, \
        result.src, \
        "", \
        "True" if result.has_ema else "False", \
        "True" if result.v2 else "False", \
        result.resolution, \
        "Checkpoint extracted successfully."


def debug_collate_fn(examples):
    input_ids = [example["input_ids"] for example in examples]
    pixel_values = [example["image"] for example in examples]
    loss_weights = torch.tensor(
        [example["res"] for example in examples], dtype=torch.float32
    )
    batch = {
        "input_ids": input_ids,
        "images": pixel_values,
        "loss_weights": loss_weights,
    }
    return batch


def debug_buckets(model_name, num_epochs, batch_size):
    print("Debug click?")
    status.textinfo = "Preparing dataset..."
    if model_name == "" or model_name is None:
        status.end()
        return "No model selected."
    args = from_file(model_name)
    if args is None:
        status.end()
        return "Invalid config."
    print("Preparing prompt dataset...")

    prompt_dataset = ClassDataset(
        args.concepts(), args.model_dir, args.resolution, False, args.disable_class_matching
    )
    inst_paths = prompt_dataset.instance_prompts
    class_paths = prompt_dataset.class_prompts
    print("Generating training dataset...")
    dataset = generate_dataset(
        model_name,
        inst_paths,
        class_paths,
        batch_size,
        debug=True,
        model_dir=args.model_dir,
    )

    placeholder = [torch.Tensor(10, 20)]
    sched_train_steps = args.num_train_epochs * dataset.__len__()

    optimizer = AdamW(
        placeholder, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if not args.use_lora and args.lr_scheduler == "dadapt_with_warmup":
        args.lora_learning_rate = args.learning_rate,
        args.lora_txt_learning_rate = args.learning_rate,

    lr_scheduler = UniversalScheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        total_training_steps=sched_train_steps,
        total_epochs=num_epochs,
        num_cycles=args.lr_cycles,
        power=args.lr_power,
        factor=args.lr_factor,
        scale_pos=args.lr_scale_pos,
        min_lr=args.learning_rate_min,
        unet_lr=args.lora_learning_rate,
        tenc_lr=args.lora_txt_learning_rate,
    )

    sampler = BucketSampler(dataset, args.train_batch_size, True)
    n_workers = 0

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=debug_collate_fn,
        pin_memory=True,
        num_workers=n_workers,
    )

    lines = []
    test_epochs = num_epochs
    sim_train_steps = test_epochs * (dataloader.__len__() // batch_size)
    print(
        f"Simulating training for {test_epochs} epochs, batch size of {batch_size}, total steps {sim_train_steps}."
    )
    for epoch in mytqdm(range(test_epochs), desc="Simulating training.", position=0):
        for step, batch in enumerate(dataloader):
            image_names = batch["images"]
            captions = batch["input_ids"]
            losses = batch["loss_weights"]
            last_lr = lr_scheduler.get_last_lr()[0]
            line = f"Epoch: {epoch}, Batch: {step}, Images: {len(image_names)}, Loss: {losses.mean()} Last LR: {last_lr}"
            print(line)
            lines.append(line)
            loss_idx = 0
            for image, caption in zip(image_names, captions):
                line = f"{image}, {caption}, {losses[loss_idx]}"
                lines.append(line)
                loss_idx += 1
            lr_scheduler.step(args.train_batch_size)
            optimizer.step()
        lr_scheduler.step(1, is_epoch=True)
    samples_dir = os.path.join(args.model_dir, "samples")
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    bucket_file = os.path.join(samples_dir, "prompts.json")
    with open(bucket_file, "w") as outfile:
        json.dump(lines, outfile, indent=4)
    try:
        del dataloader
        del dataset.tokenizer
        del dataset
        del lr_scheduler
        del optimizer
        cleanup()
    except:
        pass
    status.end()
    return "", f"Debug output saved to {bucket_file}"
