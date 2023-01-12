import glob
import json
import logging
import math
import os
import traceback

import gradio
import torch
import torch.utils.checkpoint
import torch.utils.data.dataloader
from accelerate import find_executable_batch_size
from diffusers.utils import logging as dl

from extensions.sd_dreambooth_extension.dreambooth.db_concept import Concept
from extensions.sd_dreambooth_extension.dreambooth.db_config import from_file, DreamboothConfig
from extensions.sd_dreambooth_extension.dreambooth.db_shared import status
from extensions.sd_dreambooth_extension.dreambooth.finetune_utils import ImageBuilder, PromptData, generate_dataset, \
    PromptDataset, mytqdm
from extensions.sd_dreambooth_extension.dreambooth.utils import reload_system_models, unload_system_models, get_images, \
    get_lora_models, cleanup
from modules import shared

try:
    cmd_dreambooth_models_path = shared.cmd_opts.dreambooth_models_path
except:
    cmd_dreambooth_models_path = None

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.setLevel(logging.DEBUG)
dl.set_verbosity_error()


def get_model_snapshot(config: DreamboothConfig):
    snaps_dir = os.path.join(config.model_dir, "checkpoints")
    snaps = []
    if os.path.exists(snaps_dir):
        for file in os.listdir(snaps_dir):
            if os.path.isdir(os.path.join(snaps_dir, file)):
                rev_parts = file.split("-")
                if rev_parts[0] == "checkpoint" and len(rev_parts) == 2:
                    snaps.append(rev_parts[1])
    print(f"Snaps: {snaps}")
    return snaps

def training_wizard_person(model_dir):
    return training_wizard(
        model_dir,
        is_person=True)


def training_wizard(model_dir, is_person=False):
    """
    Calculate the number of steps based on our learning rate, return the following:
    db_num_train_epochs,
    c1_num_class_images_per,
    c2_num_class_images_per,
    c3_num_class_images_per,
    db_status
    """
    if model_dir == "" or model_dir is None:
        return -1, 0, -1, 0, -1, 0, "Please select a model."
    # Load config, get total steps
    config = from_file(model_dir)

    if config is None:
        w_status = "Unable to load config."
        return 100, -1, 0, -1, 0, -1, w_status
    else:
        # Build concepts list using current settings
        concepts = config.concepts_list

        # Count the total number of images in all datasets
        total_images = 0
        counts_list = []
        max_images = 0

        # Set "base" value, which is 100 steps/image at LR of .000002
        if is_person:
            class_mult = 1
        else:
            class_mult = 0
        step_mult = 150 if is_person else 100

        for concept in concepts:
            if not os.path.exists(concept.instance_data_dir):
                print("Nonexistent instance directory.")
            else:
                concept_images = get_images(concept.instance_data_dir)
                total_images += len(concept_images)
                image_count = len(concept_images)
                print(f"Image count in {concept.instance_data_dir} is {image_count}")
                if image_count > max_images:
                    max_images = image_count
                c_dict = {
                    "concept": concept,
                    "images": image_count,
                    "classifiers": image_count * class_mult
                }
                counts_list.append(c_dict)

        c_list = []
        w_status = f"Wizard results:"
        w_status += f"<br>Num Epochs: {step_mult}"
        w_status += f"<br>Max Steps: {0}"

        for x in range(3):
            if x < len(counts_list):
                c_dict = counts_list[x]
                c_list.append(int(c_dict["classifiers"]))
                w_status += f"<br>Concept {x} Class Images: {c_dict['classifiers']}"

            else:
                c_list.append(0)

        print(w_status)

    return int(step_mult), c_list[0], c_list[1], c_list[2], w_status

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
    for i in range(sqrt_n-1, 1, -1):
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
    gradient_checkpointing: Whether to use gradient checkpointing or not.
    gradient_accumulation_steps: Number of steps to use. Set to batch size.
    mixed_precision: Mixed precision to use. BF16 will be selected if available.
    not_cache_latents: Latent caching.
    sample_batch_size: Batch size to use when creating class images.
    train_batch_size: Batch size to use when training.
    stop_text_encoder: Whether to train text encoder or not.
    use_8bit_adam: Use 8bit adam. Defaults to true.
    use_lora: Train using LORA. Better than "use CPU".
    use_ema: Train using EMA.
    msg: Stuff to show in the UI
    """
    attention = "flash_attention"
    gradient_checkpointing = False
    gradient_accumulation_steps = 1
    mixed_precision = 'fp16'
    cache_latents = True
    sample_batch_size = 1
    train_batch_size = 1
    stop_text_encoder = 0
    use_8bit_adam = True
    use_lora = False
    use_ema = False
    config = None
    if model_name == "" or model_name is None:
        print("Can't load config, specify a model name!")
    else:
        config = from_file(model_name)
    if torch.cuda.is_bf16_supported():
        mixed_precision = 'bf16'
    if config is not None:
        total_images = 0
        for concept in config.concepts_list:
            idd = concept.instance_data_dir
            if idd != "" and idd is not None and os.path.exists(idd):
                images = get_images(idd)
                total_images += len(images)
        print(f"Total images: {total_images}")
        if total_images != 0:
            best_factors = closest_factors_to_sqrt(total_images)
            largest_prime = largest_prime_factor(total_images)
            largest_factor = best_factors[0] if best_factors[0] > best_factors[1] else best_factors[1]
            smallest_factor = best_factors[0] if best_factors[0] < best_factors[1] else best_factors[1]
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
        import xformers
        import xformers.ops
        has_xformers = True
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

        msg = f"Calculated training params based on {gb}GB of VRAM:"
    except Exception as e:
        msg = f"An exception occurred calculating performance values: {e}"
        pass

    log_dict = {"Attention": attention, "Gradient Checkpointing": gradient_checkpointing,
                "Accumulation Steps": gradient_accumulation_steps, "Precision": mixed_precision,
                "Cache Latents": cache_latents, "Training Batch Size": train_batch_size,
                "Class Generation Batch Size": sample_batch_size,
                "Text Encoder Ratio": stop_text_encoder, "8Bit Adam": use_8bit_adam, "EMA": use_ema, "LORA": use_lora}
    for key in log_dict:
        msg += f"<br>{key}: {log_dict[key]}"
    return attention, gradient_checkpointing, gradient_accumulation_steps, mixed_precision, cache_latents, \
        sample_batch_size, train_batch_size, stop_text_encoder, use_8bit_adam, use_lora, use_ema, msg



def ui_samples(model_dir: str,
               save_sample_prompt: str,
               num_samples: int = 1,
               sample_batch_size: int = 1,
               lora_model_path: str = "",
               lora_rank: float = 1,
               lora_weight: float = 1,
               lora_txt_weight: float = 1,
               negative_prompt: str = "",
               seed: int = -1,
               steps: int = 60,
               scale: float = 7.5
               ):

    if sample_batch_size > num_samples:
        sample_batch_size = num_samples
    @find_executable_batch_size(starting_batch_size=sample_batch_size)
    def sample_loop(batch_size):
        if model_dir is None or model_dir == "":
            return "Please select a model."
        config = from_file(model_dir)
        msg = f"Generated {num_samples} sample(s)."
        images = []
        prompts_out = []
        if save_sample_prompt is None:
            msg = "Please provide a sample prompt."
            print(msg)
            return None, msg
        try:
            unload_system_models()
            print(f"Loading model from {config.model_dir}.")
            status.textinfo = "Loading diffusion model..."
            img_builder = ImageBuilder(
                config,
                False,
                lora_model_path,
                lora_weight,
                lora_txt_weight,
                batch_size)
            status.textinfo = f"Generating sample image for model {config.model_name}..."
            pd = PromptData()
            pd.steps = steps
            pd.prompt = save_sample_prompt
            pd.negative_prompt = negative_prompt
            pd.scale = scale
            pd.seed = seed
            prompts = [pd] * batch_size
            pbar = mytqdm("Generating samples")
            while len(images) < num_samples:
                prompts_out.append(save_sample_prompt)
                out_images = img_builder.generate_images(prompts)
                for img in out_images:
                    if len(images) < num_samples:
                        pbar.update()
                        images.append(img)
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
        reload_system_models()
        print(f"Returning {len(images)} samples.")
        prompt_str = "<br>".join(prompts_out)
        return images, prompt_str, msg
    return sample_loop()

def load_params(model_dir):
    data = from_file(model_dir)
    concepts = []
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
            if key == "concepts_list":
                concepts = value
            else:
                if key == "pretrained_model_name_or_path":
                    key = "model_path"
                ui_dict[f"db_{key}"] = value
                msg = "Loaded config."

    ui_concept_list = concepts if concepts is not None else []
    if len(ui_concept_list) < 3:
        while len(ui_concept_list) < 3:
            ui_concept_list.append(Concept())
    c_idx = 1
    for ui_concept in ui_concept_list:
        if c_idx > 3:
            break

        for key in sorted(ui_concept):
            ui_dict[f"c{c_idx}_{key}"] = ui_concept[key]
        c_idx += 1
    ui_dict["db_status"] = msg
    ui_keys = ["db_attention",
               "db_cache_latents",
               "db_center_crop",
               "db_clip_skip",
               "db_concepts_path",
               "db_custom_model_name",
               "db_epoch_pause_frequency",
               "db_epoch_pause_time",
               "db_gradient_accumulation_steps",
               "db_gradient_checkpointing",
               "db_gradient_set_to_none",
               "db_graph_smoothing",
               "db_half_model",
               "db_hflip",
               "db_learning_rate",
               "db_learning_rate_min",
               "db_lora_learning_rate",
               "db_lora_model_name",
               "db_lora_model_rank",
               "db_lora_txt_learning_rate",
               "db_lora_txt_weight",
               "db_lora_weight",
               "db_lr_cycles",
               "db_lr_factor",
               "db_lr_power",
               "db_lr_scale_pos",
               "db_lr_scheduler",
               "db_lr_warmup_steps",
               "db_max_token_length",
               "db_mixed_precision",
               "db_adamw_weight_decay",
               "db_num_train_epochs",
               "db_pad_tokens",
               "db_pretrained_vae_name_or_path",
               "db_prior_loss_weight",
               "db_resolution",
               "db_sample_batch_size",
               "db_sanity_prompt",
               "db_sanity_seed",
               "db_save_ckpt_after",
               "db_save_ckpt_cancel",
               "db_save_ckpt_during",
               "db_save_embedding_every",
               "db_save_lora_after",
               "db_save_lora_cancel",
               "db_save_lora_during",
               "db_save_preview_every",
               "db_save_safetensors",
               "db_save_state_after",
               "db_save_state_cancel",
               "db_save_state_during",
               "db_shuffle_tags",
               "db_snapshot",
               "db_train_batch_size",
               "db_train_imagic",
               "db_stop_text_encoder",
               "db_use_8bit_adam",
               "db_use_concepts",
               "db_use_ema",
               "db_use_lora",
               "db_use_subdir",
               "c1_class_data_dir", "c1_class_guidance_scale", "c1_class_infer_steps",
               "c1_class_negative_prompt", "c1_class_prompt", "c1_class_token",
               "c1_instance_data_dir", "c1_instance_prompt", "c1_instance_token", "c1_n_save_sample",
               "c1_num_class_images", "c1_num_class_images_per", "c1_sample_seed", "c1_save_guidance_scale", "c1_save_infer_steps",
               "c1_save_sample_negative_prompt", "c1_save_sample_prompt", "c1_save_sample_template",
               "c2_class_data_dir",
               "c2_class_guidance_scale", "c2_class_infer_steps", "c2_class_negative_prompt", "c2_class_prompt",
               "c2_class_token", "c2_instance_data_dir", "c2_instance_prompt",
               "c2_instance_token", "c2_n_save_sample", "c2_num_class_images", "c2_num_class_images_per", "c2_sample_seed",
               "c2_save_guidance_scale", "c2_save_infer_steps", "c2_save_sample_negative_prompt",
               "c2_save_sample_prompt", "c2_save_sample_template", "c3_class_data_dir", "c3_class_guidance_scale",
               "c3_class_infer_steps", "c3_class_negative_prompt", "c3_class_prompt", "c3_class_token",
               "c3_instance_data_dir", "c3_instance_prompt", "c3_instance_token",
               "c3_n_save_sample", "c3_num_class_images", "c3_num_class_images_per", "c3_sample_seed", "c3_save_guidance_scale",
               "c3_save_infer_steps", "c3_save_sample_negative_prompt", "c3_save_sample_prompt",
               "c3_save_sample_template", "db_status"]
    output = []
    for key in ui_keys:
        if key in ui_dict:
            if key == "db_v2" or key == "db_has_ema":
                output.append("True" if ui_dict[key] else "False")
            else:
                output.append(ui_dict[key])
        else:
            if 'epoch' in key:
                output.append(0)
            else:
                output.append(None)
    print(f"Returning {output}")
    return output


def load_model_params(model_name):
    """
    @param model_name: The name of the model to load.
    @return:
    db_model_path: The full path to the model directory
    db_revision: The current revision of the model
    db_v2: If the model requires a v2 config/compilation
    db_has_ema: Was the model extracted with EMA weights
    db_src: The source checkpoint that weights were extracted from or hub URL
    db_scheduler: Scheduler used for this model
    db_model_snapshots: A gradio dropdown containing the available snapshots for the model
    db_outcome: The result of loading model params
    """
    data = from_file(model_name)
    db_model_snapshots = gradio.update(choices=[], value="")
    if data is None:
        print("Can't load config!")
        msg = f"Error loading model params: '{model_name}'."
        return "", "", "", "", "", "", db_model_snapshots, msg
    else:
        snaps = get_model_snapshot(data)
        snap_selection = data.revision if str(data.revision) in snaps else ""
        snaps.insert(0, "")
        db_model_snapshots = gradio.update(choices=snaps, value=snap_selection)

        msg = f"Selected model: '{model_name}'."
        return data.model_dir, \
            data.revision, \
            data.epoch, \
            "True" if data.v2 else "False", \
            "True" if data.has_ema else "False", \
            data.src, \
            data.scheduler, \
            db_model_snapshots, \
            msg


def start_training(model_dir: str, use_txt2img: bool = True):
    """

    @param model_dir: The directory containing the dreambooth model/config
    @param use_txt2img: Whether to use txt2img or diffusion pipeline for image generation.
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
        lora_model_name = gradio.update(visible=True)
        return lora_model_name, 0, 0, [], msg
    config = from_file(model_dir)

    # Clear pretrained VAE Name if applicable
    if config.pretrained_vae_name_or_path == "":
        config.pretrained_vae_name_or_path = None

    msg = None
    if config.attention == "xformers":
        if config.mixed_precision == "no":
            msg = "Using xformers, please set mixed precision to 'fp16' or 'bf16' to continue."
    if not len(config.concepts_list):
        msg = "Please configure some concepts."
    if not os.path.exists(config.pretrained_model_name_or_path):
        msg = "Invalid training data directory."
    if config.pretrained_vae_name_or_path != "" and config.pretrained_vae_name_or_path is not None:
        if not os.path.exists(config.pretrained_vae_name_or_path):
            msg = "Invalid Pretrained VAE Path."
    if config.resolution <= 0:
        msg = "Invalid resolution."

    if msg:
        print(msg)
        lora_model_name = gradio.update(visible=True)
        return lora_model_name, 0, 0, [], msg

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
            from extensions.sd_dreambooth_extension.dreambooth.train_imagic import train_imagic
            result = train_imagic(config)
        else:
            status.textinfo = "Initializing dreambooth training..."
            print(status.textinfo)
            from extensions.sd_dreambooth_extension.dreambooth.train_dreambooth import main
            result = main(config, use_txt2img=use_txt2img)

        config = result.config
        images = result.samples
        if config.revision != total_steps:
            config.save()
        else:
            log_dir = os.path.join(config.model_dir, "logging", "dreambooth", "*")
            list_of_files = glob.glob(log_dir)
            latest_file = max(list_of_files, key=os.path.getmtime)
            print(f"No training was completed, deleting log: {latest_file}")
            os.remove(latest_file)
        total_steps = config.revision
        res = f"Training {'interrupted' if status.interrupted else 'finished'}. " \
              f"Total lifetime steps: {total_steps} \n"
    except Exception as e:
        res = f"Exception training model: '{e}'."
        traceback.print_exc()
        pass

    cleanup()
    reload_system_models()
    if config.lora_model_name != "" and config.lora_model_name is not None:
        lora_model_name = f"{config.model_name}_{total_steps}.pt"
    dirs = get_lora_models()
    lora_model_name = gradio.Dropdown.update(choices=sorted(dirs), value=config.lora_model_name)
    return lora_model_name, total_steps, config.epoch, images, res


def ui_classifiers(model_name: str,
                   use_txt2img: bool):
    """
    UI method for generating class images.
    @param model_name: The model to generate classes for.
    @param use_txt2img: Use txt2image when generating concepts.
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
    if not len(config.concepts_list):
        msg = "Please configure some concepts."
    if not os.path.exists(config.pretrained_model_name_or_path):
        msg = "Invalid training data directory."
    if config.pretrained_vae_name_or_path != "" and config.pretrained_vae_name_or_path is not None:
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
        from extensions.sd_dreambooth_extension.dreambooth.train_dreambooth import generate_classifiers
        print("Generating class images...")
        unload_system_models()
        count, images = generate_classifiers(config, use_txt2img=use_txt2img, ui=True)
        reload_system_models()
        msg = f"Generated {count} class images."
    except Exception as e:
        msg = f"Exception generating concepts: {str(e)}"
        traceback.print_exc()
        status.job_no = status.job_count
        status.textinfo = msg
    return images, msg

def collate_fn(examples):
    return examples[0]

def debug_buckets(model_name):
    print("Debug click?")
    status.textinfo = "Preparing dataset..."
    if model_name == "" or model_name is None:
        return "No model selected."
    args = from_file(model_name)
    if args is None:
        return "Invalid config."
    print("Preparing prompt dataset...")
    prompt_dataset = PromptDataset(args.concepts_list, args.model_dir, args.resolution)
    inst_paths = prompt_dataset.instance_paths
    class_paths = prompt_dataset.class_paths
    print("Generating training dataset...")
    dataset = generate_dataset(model_name, inst_paths, class_paths, 10, debug=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    lines = []
    test_epochs = 10
    print(f"Simulating training for {test_epochs} epochs.")
    for epoch in mytqdm(range(test_epochs), desc="Simulating training."):
        for step, batch in enumerate(dataloader):
            image_names = batch["images"]
            captions = batch["input_ids"]
            losses = batch["loss_weight"]
            res = batch["res"]
            line = f"Epoch: {epoch}, Batch: {step}, Images: {len(image_names)}, Res: {res}, Loss: {losses.mean()}"
            print(line)
            lines.append(line)
            for image, caption in zip(image_names, captions):
                line = f"{image}, {caption}, {losses}"
                lines.append(line)
    samples_dir = os.path.join(args.model_dir, "samples")
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    bucket_file = os.path.join(samples_dir, "prompts.json")
    with open(bucket_file, "w") as outfile:
        json.dump(lines, outfile, indent=4)
    status.end()
    return f"Debug output saved to {bucket_file}"


