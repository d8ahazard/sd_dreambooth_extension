import gc
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

import gradio
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
        scheduler: str = "UniPCMultistep"
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
        return gradio.update(visible=False), "", "", "", "", "", db_model_snapshots, msg
    else:
        snaps = get_model_snapshots(config)
        snap_selection = config.revision if str(config.revision) in snaps else ""
        db_model_snapshots = gr_update(choices=snaps, value=snap_selection)

        loras = get_lora_models(config)
        db_lora_models = gr_update(choices=loras)
        msg = f"Selected model: '{model_name}'."
        src_name = os.path.basename(config.src)
        # Strip the extension
        src_name = os.path.splitext(src_name)[0]
        return (
            gradio.update(visible=True),
            os.path.basename(config.model_dir),
            config.revision,
            config.epoch,
            config.model_type,
            "True" if config.has_ema and not config.use_lora else "False",
            src_name,
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
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
        lora_model_name = f"{config.model_name}_{total_steps}.safetensors"
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
        model_type="v1x"
):
    if not model_type:
        model_type = "v1x"
    printm("Extracting model.")
    res = 512
    is_512 = model_type == "v1x"
    if model_type == "v1x" or model_type=="v2x-512":
        res = 512
    elif model_type == "v2x":
        res = 768
    elif model_type == "SDXL":
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


def create_workspace(
    project_name: str,
    source_model_name: str,
    source_model_type: str="v1x"
):
    err_msg = None
    if project_name is None or project_name == "":
        print("No project name.")
        err_msg = "Please enter a project name"
    if source_model_name is None or source_model_name == "":
        print("No model name.")
        err_msg = "Please select a model"
    if source_model_type is None or source_model_type == "":
        print("No model type.")
        err_msg = "Please select a model type"
    if err_msg:
        return "", "", "", 0, 0, "", "", "", "", 0, "", err_msg
    finetune_models_path = os.path.join(shared.models_path, "finetune")
    diffusers_models_path = os.path.join(shared.models_path, "diffusers")
    extracted_path = os.path.join(diffusers_models_path, os.path.basename(source_model_name))
    for path in [finetune_models_path, diffusers_models_path, extracted_path]:
        if not os.path.exists(path):
            os.makedirs(path)
        print(f"Extracting {source_model_name} to {extracted_path}")
    if not extract_checkpoint(new_model_name=os.path.basename(source_model_name),
                           checkpoint_file=source_model_name,
                           train_unfrozen=False,
                           image_size=512,
                           model_type=source_model_type,
                           out_dir=extracted_path):
        print(f"Error extracting {source_model_name} to {extracted_path}")
        return "", "", "", 0, 0, "", "", "", "", 0, "", "Error extracting checkpoint"
    return gr_update(choices=sorted(get_db_models()), value=project_name), extracted_path, "", "", 0, 0, "", "", "", "", 0, "", "Workspace created successfully."


