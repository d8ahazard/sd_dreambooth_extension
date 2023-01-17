import gc
import json
import os
import random
import re
import traceback
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

import gradio
import numpy as np
import tensorflow
import torch
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from diffusers import DiffusionPipeline, AutoencoderKL
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import CLIPTextModel, AutoTokenizer

from extensions.sd_dreambooth_extension.dreambooth import db_shared
from extensions.sd_dreambooth_extension.dreambooth.db_concept import Concept
from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig, from_file
from extensions.sd_dreambooth_extension.dreambooth.db_shared import status
from extensions.sd_dreambooth_extension.dreambooth.prompt_data import PromptData
from extensions.sd_dreambooth_extension.dreambooth.utils import cleanup, get_checkpoint_match, get_images, db_save_image
from extensions.sd_dreambooth_extension.lora_diffusion.lora import apply_lora_weights
from modules import shared, devices, sd_models, sd_hijack, prompt_parser, lowvram
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessing, Processed, \
    get_fixed_seed, create_infotext, decode_first_stage
from modules.sd_hijack import model_hijack


class mytqdm(tqdm):
    def __init__(self, iterable: Iterable = None, **kwargs):
        self.update_ui = True
        if "total" in kwargs:
            total = kwargs["total"]
            if total is not None:
                db_shared.status.job_count = kwargs["total"]
        if "desc" in kwargs:
            desc = kwargs["desc"]
            if desc is not None:
                if "." not in desc and ":" not in desc:
                    desc = f"{desc}:"
                db_shared.status.textinfo = desc
        super().__init__(iterable=iterable, **kwargs)

    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""
        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable
        db_shared.status.job_count = len(iterable)

        # If the bar is disabled, then just walk the iterable
        # (note: keep this check outside the loop for performance)
        if self.disable:
            for obj in iterable:
                yield obj
            return

        mininterval = self.mininterval
        last_print_t = self.last_print_t
        last_print_n = self.last_print_n
        min_start_t = self.start_t + self.delay
        n = self.n
        time = self._time

        try:
            for obj in iterable:
                yield obj
                # Update and possibly print the progressbar.
                # Note: does not call self.update(1) for speed optimisation.
                n += 1

                if n - last_print_n >= self.miniters:
                    cur_t = time()
                    dt = cur_t - last_print_t
                    if dt >= mininterval and cur_t >= min_start_t:
                        self.update(n - last_print_n)
                        last_print_n = self.last_print_n
                        last_print_t = self.last_print_t
        finally:
            self.n = n
            self.close()

    def update(self, n=1):
        if self.update_ui:
            db_shared.status.job_no += n
            if db_shared.status.job_no > db_shared.status.job_count:
                db_shared.status.job_no = db_shared.status.job_count
        super().update(n)

    def reset(self, total=None):
        if total is not None and self.update_ui:
            db_shared.status.job_no = 0
            db_shared.status.job_count = total
        super().reset(total)

    def set_description(self, desc=None, refresh=True):
        if self.update_ui:
            db_shared.status.textinfo = desc
        super().set_description(desc, refresh)
    def pause_ui(self):
        self.update_ui = False

    def unpause_ui(self):
        self.update_ui = True

class CustomAccelerator(Accelerator):
    def __init__(self, logfile, *args, **kwargs):
        self.logfile = logfile
        self.summary_writer = tensorflow.summary.create_file_writer(self.logfile)
        super().__init__(*args, **kwargs)
    def _log(self, step, metrics, prefix=''):
        with self.summary_writer.as_default():
            for name, value in metrics.items():
                tensorflow.summary.scalar(name, value, step=step)
            self.summary_writer.flush()

class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0
        self.collected_params = []

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)
        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        db_shared.torch_gc()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    # From CompVis LitEMA implementation
    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

        del self.collected_params
        gc.collect()

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
            dtype: Floating point-type for the stuff.
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

class FilenameTextGetter:
    """Adapted from modules.textual_inversion.dataset.PersonalizedBase to get caption for image."""

    re_numbers_at_start = re.compile(r"^[-\d]+\s*")

    def __init__(self, shuffle_tags=False):
        self.re_word = re.compile(db_shared.dataset_filename_word_regex) if len(
            db_shared.dataset_filename_word_regex) > 0 else None
        self.shuffle_tags = shuffle_tags

    def read_text(self, img_path):
        text_filename = os.path.splitext(img_path)[0] + ".txt"
        filename = os.path.basename(img_path)

        if os.path.exists(text_filename):
            with open(text_filename, "r", encoding="utf8") as file:
                filename_text = file.read().strip()
        else:
            filename_text = os.path.splitext(filename)[0]
            filename_text = re.sub(self.re_numbers_at_start, '', filename_text)
            if self.re_word:
                tokens = self.re_word.findall(filename_text)
                filename_text = (db_shared.dataset_filename_join_string or "").join(tokens)

        filename_text = filename_text.replace("\\", "")  # work with \(franchies\)
        return filename_text

    def create_text(self, text_template, filename_text, instance_token, class_token, is_class=True):
        # If we are creating text for a class image and it has our instance token in it, remove/replace it
        class_tokens = [f"a {class_token}", f"the {class_token}", f"an {class_token}", class_token]
        if instance_token != "" and class_token != "":
            if is_class and instance_token in filename_text:
                if class_token in filename_text:
                    filename_text = filename_text.replace(instance_token, "")
                    filename_text = filename_text.replace("  ", " ")
                else:
                    filename_text = filename_text.replace(instance_token, class_token)

            if not is_class:
                if class_token in filename_text:
                    # Do nothing if we already have class and instance in string
                    if instance_token in filename_text:
                        pass
                    # Otherwise, substitute class tokens for the base token
                    else:
                        for token in class_tokens:
                            if token in filename_text:
                                filename_text = filename_text.replace(token, f"{class_token}")
                    # Now, replace class with instance + class tokens
                    filename_text = filename_text.replace(class_token, f"{instance_token} {class_token}")
                else:
                    # If class is not in the string, check if instance is
                    if instance_token in filename_text:
                        filename_text = filename_text.replace(instance_token, f"{instance_token} {class_token}")
                    else:
                        # Description only, insert both at the front?
                        filename_text = f"{instance_token} {class_token}, {filename_text}"

        # Append the filename text to the template first...THEN shuffle it all.
        output = text_template.replace("[filewords]", filename_text)
        # Remove underscores, double-spaces, and other characters that will cause issues.
        output.replace("_", " ")
        output.replace("  ", " ")
        strip_chars = ["(", ")", "/", "\\", ":", "[", "]"]
        for s_char in strip_chars:
            output.replace(s_char, "")

        tags = output.split(',')
                
        if self.shuffle_tags and len(tags) > 2:
            first_tag = tags.pop(0)
            random.shuffle(tags)
            tags.insert(0, first_tag)

        if is_class:
            if class_token and class_token not in tags:
                tags.insert(0, class_token)
        else:
            if instance_token and instance_token not in tags:
                tags.insert(0, instance_token)

        output = ','.join(tags)
        return output


class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, concepts: [Concept], model_dir: str, max_width:int):
        # Existing training image data
        self.instance_prompts = []
        # Existing class image data
        self.class_prompts = []
        # Data for new prompts to generate
        self.new_prompts = {}
        self.required_prompts = 0
        # Calculate minimum width
        min_width = (int(max_width * 0.28125) // 64) * 64

        # Thingy to build prompts
        text_getter = FilenameTextGetter(False)

        # Create available resolutions
        bucket_resos = make_bucket_resolutions(max_width, min_width)

        c_idx = 0

        for concept in concepts:
            instance_dir = concept.instance_data_dir
            if not concept.is_valid():
                continue
            class_dir = concept.class_data_dir
            instance_prompts = {}

            # Filter empty class dir and set/create if necessary
            if class_dir == "" or class_dir is None or class_dir == db_shared.script_path:
                class_dir = os.path.join(model_dir, f"classifiers_{c_idx}")
            class_dir = Path(class_dir)
            class_dir.mkdir(parents=True, exist_ok=True)

            status.textinfo = "Sorting images..."
            # Sort existing prompts
            if instance_dir:
                instance_prompts = sort_prompts(concept, text_getter, instance_dir, bucket_resos, False)
            if concept.num_class_images_per > 0 and class_dir:
                class_prompts = sort_prompts(concept, text_getter, class_dir, bucket_resos, True)
            else:
                class_prompts = {}
            idx = 0
            matched_resos = []
            new_prompts = []
            print(f"Concept requires {concept.num_class_images_per} class images per instance image.")
            for res, prompts in mytqdm(instance_prompts.items(), desc="Sorting instance prompts"):
                if len(prompts) == 0:
                    print(f"No prompts for res {re}")
                    continue
                self.instance_prompts.extend(prompts)
                matched_resos.append((idx, res))
                idx += 1

                if concept.num_class_images_per > 0:
                    class_check = class_prompts[res] if res in class_prompts.keys() else []
                    num_classes = len(prompts) * concept.num_class_images_per
                    if len(class_check) >= num_classes:
                        class_check = random.sample(class_check, num_classes)
                    else:
                        missing_prompts = num_classes - len(class_check)
                        while missing_prompts > 0:
                            rand = random.choice(prompts)
                            instance_prompt = rand.prompt
                            sample_prompt = text_getter.create_text(concept.class_prompt, instance_prompt, rand.instance_token,rand.class_token, True)
                            pd = PromptData(
                                prompt=sample_prompt,
                                prompt_tokens=[(concept.instance_token, concept.class_token)],
                                negative_prompt=concept.class_negative_prompt,
                                instance_token=concept.instance_token,
                                class_token=concept.class_token,
                                steps=concept.class_infer_steps,
                                scale=concept.class_guidance_scale,
                                out_dir=class_dir,
                                seed=-1,
                                resolution=res
                            )
                            new_prompts.append(pd)
                            self.required_prompts += 1
                            missing_prompts -= 1
                    self.class_prompts.extend(class_check)

                if len(new_prompts):
                    if res in self.new_prompts:
                        self.new_prompts[res].extend(new_prompts)
                    else:
                        self.new_prompts[res] = new_prompts
            c_idx += 1

        if self.required_prompts > 0:
            print(f"We need a total of {self.required_prompts} class images.")

    def __len__(self) -> int:
        return self.required_prompts

    def __getitem__(self, index) -> PromptData:
        res_index = 0
        for res, prompt_datas in self.new_prompts.items():
            for p in range(len(prompt_datas)):
                if res_index == index:
                    return prompt_datas[p]
                res_index += 1
        print(f"Invalid index: {index}/{self.required_prompts}")
        return None

class ImageBuilder:
    def __init__(self, config: DreamboothConfig, use_txt2img: bool, lora_model: str = None, lora_weight: float = 1,
                 lora_txt_weight: float = 1, batch_size: int = 1, accelerator: Accelerator = None):
        self.image_pipe = None
        self.txt_pipe = None
        self.resolution = config.resolution
        self.last_model = None
        self.batch_size = batch_size
        config_src = config.src
        if not os.path.exists(config_src):
            alt_src = os.path.join(db_shared.dreambooth_models_path, config_src)
            if os.path.exists(alt_src):
                config_src = alt_src
        if not os.path.exists(config_src) and use_txt2img:
            print("Source model is from hub, can't use txt2image.")
            use_txt2img = False

        self.use_txt2img = use_txt2img
        self.del_accelerator = False

        if not self.use_txt2img:
            self.accelerator = accelerator
            if accelerator is None:
                try:
                    accelerator = Accelerator(
                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                        mixed_precision=config.mixed_precision,
                        log_with="tensorboard",
                        logging_dir=os.path.join(config.model_dir, "logging")
                    )
                    self.accelerator = accelerator
                    self.del_accelerator = True
                except Exception as e:
                    if "AcceleratorState" in str(e):
                        msg = "Change in precision detected, please restart the webUI entirely to use new precision."
                    else:
                        msg = f"Exception initializing accelerator: {e}"
                    print(msg)
            torch_dtype = torch.float16 if shared.device.type == "cuda" else torch.float32

            self.image_pipe = DiffusionPipeline.from_pretrained(
                config.pretrained_model_name_or_path,
                vae=AutoencoderKL.from_pretrained(
                    config.pretrained_vae_name_or_path or config.pretrained_model_name_or_path,
                    subfolder=None if config.pretrained_vae_name_or_path else "vae",
                    revision=config.revision,
                    torch_dtype=torch_dtype
                ),
                torch_dtype=torch_dtype,
                requires_safety_checker=False,
                safety_checker=None,
                revision=config.revision
            )

            self.image_pipe.to(accelerator.device)
            new_hotness = os.path.join(config.model_dir, "checkpoints", f"checkpoint-{config.revision}")
            if os.path.exists(new_hotness):
                accelerator.print(f"Resuming from checkpoint {new_hotness}")
                no_safe = db_shared.stop_safe_unpickle()
                accelerator.load_state(new_hotness)
                if no_safe:
                    db_shared.start_safe_unpickle()
            if config.use_lora and lora_model is not None and lora_model != "":
                apply_lora_weights(self.image_pipe.unet, self.image_pipe.text_encoder, config, is_ui=True)
        else:
            current_model = sd_models.select_checkpoint()
            new_model_info = get_checkpoint_match(config.src)
            if new_model_info is not None and current_model is not None:
                if new_model_info.model_name != current_model.model_name:
                    self.last_model = current_model
                    print(f"Loading model: {new_model_info.model_name}")
                    sd_models.load_model(new_model_info)
            if new_model_info is not None and current_model is None:
                sd_models.load_model(new_model_info)
            shared.sd_model.to(shared.device)


    def generate_images(self, prompt_data: List[PromptData]) -> [Image]:
        positive_prompts = []
        negative_prompts = []
        seed = -1
        scale = 7.5
        steps = 60
        width = self.resolution
        height = self.resolution
        for prompt in prompt_data:
            positive_prompts.append(prompt.prompt)
            negative_prompts.append(prompt.negative_prompt)
            scale = prompt.scale
            steps = prompt.steps
            seed = prompt.seed
            width, height = prompt.resolution

        if self.use_txt2img:
            p = StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                prompt=positive_prompts,
                negative_prompt=negative_prompts,
                batch_size=self.batch_size,
                steps=steps,
                cfg_scale=scale,
                width=width,
                height=height,
                do_not_save_grid=True,
                do_not_save_samples=True,
                do_not_reload_embeddings=True
            )
            processed = process_txt2img(p)
            p.close()
            output = processed
        else:
            with self.accelerator.autocast(), torch.inference_mode():
                if seed is None or seed == '' or seed == -1:
                    seed = int(random.randrange(21474836147))
                g_cuda = torch.Generator(device=self.accelerator.device).manual_seed(seed)
                output = self.image_pipe(
                    positive_prompts,
                    num_inference_steps=steps,
                    guidance_scale=scale,
                    height=height,
                    width=width,
                    generator=g_cuda,
                    negative_prompt=negative_prompts).images

        return output

    def unload(self, is_ui):
        # If we have an image pipe, delete it
        if self.image_pipe is not None:
            del self.image_pipe
        if self.del_accelerator:
            del self.accelerator
        # If there was a model loaded already, reload it
        if self.last_model is not None and not is_ui:
            sd_models.load_model(self.last_model)


def get_dim(filename, max_res):
    with Image.open(filename) as im:
        width, height = im.size
        exif = im.getexif()
        if exif:
            orientation = exif.get(274)
            if orientation == 3 or orientation == 6:
                width, height = height, width
        if width > max_res or height > max_res:
            aspect_ratio = width / height
            if width > height:
                width = max_res
                height = int(max_res / aspect_ratio)
            else:
                height = max_res
                width = int(max_res * aspect_ratio)
        return width, height


def sort_prompts(concept: Concept, text_getter: FilenameTextGetter, img_dir: str, bucket_resos: List[Tuple[int, int]],
                 is_class: bool) -> Dict[Tuple[int, int], PromptData]:
    prompts = {}
    images = get_images(img_dir)
    max_dim = 0
    for (w, h) in bucket_resos:
        if w > max_dim:
            max_dim = w
        if h > max_dim:
            max_dim = h
    _, dirr = os.path.split(img_dir)
    for img in mytqdm(images, desc=f"Pre-processing {dirr}"):
        # Get prompt
        text = text_getter.read_text(img)
        prompt = text_getter.create_text(concept.class_prompt, text, concept.instance_token, concept.class_token, is_class)
        w, h = get_dim(img, max_dim)
        reso = closest_resolution(w, h, bucket_resos)
        prompt_list = prompts[reso] if reso in prompts else []
        pd = PromptData(prompt, prompt_to_tags(prompt), None, concept.instance_token, concept.class_token, img, resolution=reso)
        prompt_list.append(pd)
        prompts[reso] = prompt_list
            
    return dict(sorted(prompts.items()))

def prompt_to_tags(src_prompt: str, instance_token: str = None, class_token: str = None):
    src_tags = src_prompt.split(',')
    if class_token:
        conjunctions = ['a ', 'an ', 'the ']
        src_tags = [tag.replace(conjunction + class_token, '') for tag in src_tags for conjunction in conjunctions]
    if class_token and instance_token:
        src_tags = [tag.replace(instance_token, '').replace(class_token, '') for tag in src_tags]
    src_tags = [' '.join(tag.split()) for tag in src_tags]
    src_tags = [tag.strip() for tag in src_tags if tag]
    return src_tags


def compare_prompts(src_prompt: str, check_prompt: str, tokens: [Tuple[str, str]]):
    src_tags = src_prompt.split(',')
    check_tags = check_prompt.split(',')
    conjunctions = ['a ', 'an ', 'the ']
    # Loop pairs of tokens
    for token_pair in tokens:
        # Filter conjunctions
        for conjunction in conjunctions:
            src_tags = [tag.replace(conjunction + token_pair[1], '') for tag in src_tags]
            check_tags = [tag.replace(conjunction + token_pair[1], '') for tag in check_tags]
        # Remove individual tags
        src_tags = [tag.replace(token_pair[0], '').replace(token_pair[1], '') for tag in src_tags]
        check_tags = [tag.replace(token_pair[0], '').replace(token_pair[1], '') for tag in check_tags]

    # Strip double spaces
    src_tags = [' '.join(tag.split()) for tag in src_tags]
    check_tags = [' '.join(tag.split()) for tag in check_tags]

    # Strip
    src_tags = [tag.strip() for tag in src_tags]
    check_tags = [tag.strip() for tag in check_tags]

    # Remove empty tags
    src_tags = [tag for tag in src_tags if tag]
    check_tags = [tag for tag in check_tags if tag]
    return set(src_tags) == set(check_tags)




def make_bucket_resolutions(max_size, min_size=256, divisible=64):
    resos = set()

    w = max_size
    while w > min_size:
        h = max_size
        while h > min_size:
            resos.add((w, h))
            resos.add((h, w))
            h -= divisible
        w -= divisible

    resos = list(resos)
    resos.sort()
    return resos


def closest_resolution(width, height, resos):
    def distance(reso):
        w, h = reso
        if w > width or h > height:
            return float("inf")
        return (w - width) ** 2 + (h - height) ** 2

    return min(resos, key=distance)

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

def get_captions(concept: Concept, paths: List[str], is_class: bool = True):
    captions = []
    text_getter = FilenameTextGetter()
    src_dir = concept.instance_data_dir if not is_class else concept.class_data_dir
    for img_path in paths:
        if src_dir not in str(img_path):
            continue
        cap_for_img = text_getter.read_text(img_path)
        final_caption = text_getter.create_text(concept.instance_prompt, cap_for_img, concept.instance_token,
                                                concept.class_token, is_class)
        captions.append(final_caption)

    return list(zip(paths, captions))


def process_txt2img(p: StableDiffusionProcessing) -> [Image]:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if type(p.prompt) == list:
        assert (len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    sd_hijack.model_hijack.clear_comments()

    comments = {}

    if type(p.prompt) == list:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
    else:
        p.all_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

    if type(p.negative_prompt) == list:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in
                                  p.negative_prompt]
    else:
        p.all_negative_prompts = p.batch_size * p.n_iter * [
            shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

    if type(seed) == list:
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    if type(subseed) == list:
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

    with open(os.path.join(db_shared.script_path, "params.txt"), "w", encoding="utf8") as file:
        processed = Processed(p, [], p.seed, "")
        file.write(processed.infotext(p, 0))

    infotexts = []
    output_images = []

    with torch.no_grad(), p.sd_model.ema_scope():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

        if status.job_count == -1:
            status.job_count = p.n_iter

        for n in range(p.n_iter):
            if status.skipped:
                status.skipped = False

            if status.interrupted:
                break

            prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if len(prompts) == 0:
                break

            with devices.autocast():
                uc = prompt_parser.get_learned_conditioning(shared.sd_model, negative_prompts, p.steps)
                c = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, prompts, p.steps)

            if len(model_hijack.comments) > 0:
                for comment in model_hijack.comments:
                    comments[comment] = 1

            if p.n_iter > 1:
                status.job = f"Batch {n + 1} out of {p.n_iter}"

            with devices.autocast():
                samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds,
                                        subseed_strength=p.subseed_strength, prompts=prompts)

            x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i + 1].to(dtype=devices.dtype_vae))[0].cpu()
                              for i in range(samples_ddim.size(0))]
            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim

            if db_shared.lowvram or db_shared.medvram:
                lowvram.send_everything_to_cpu()

            devices.torch_gc()

            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                image = Image.fromarray(x_sample)

                text = infotext(n, i)
                infotexts.append(text)
                image.info["parameters"] = text
                output_images.append(image)

            del x_samples_ddim

            devices.torch_gc()

            status.nextjob()

        p.color_corrections = None

    devices.torch_gc()

    return output_images


def generate_prompts(model_dir):
    status.job_count = 4
    from extensions.sd_dreambooth_extension.dreambooth.SuperDataset import SuperDataset
    if model_dir is None or model_dir == "":
        return "Please select a model."
    config = from_file(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(config.pretrained_model_name_or_path, "tokenizer"),
        revision=config.revision,
        use_fast=False,
    )
    status.job_no = 1
    status.textinfo = "Building dataset from existing files..."
    train_dataset = SuperDataset(
        concepts_list=config.concepts_list,
        tokenizer=tokenizer,
        size=config.resolution,
        center_crop=config.center_crop,
        lifetime_steps=config.revision,
        pad_tokens=config.pad_tokens,
        hflip=config.hflip,
        max_token_length=config.max_token_length,
        shuffle_tags=config.shuffle_tags
    )

    output = {"instance_prompts": [], "existing_class_prompts": [], "new_class_prompts": [], "sample_prompts": []}
    status.job_no = 2
    status.textinfo = "Appending instance and class prompts from existing files..."
    for i in range(train_dataset.__len__()):
        item = train_dataset.__getitem__(i)
        output["instance_prompts"].append(item["instance_prompt"])
        if "class_prompt" in item:
            output["existing_class_prompts"].append(item["class_prompt"])
    sample_prompts = train_dataset.get_sample_prompts()
    for prompt in sample_prompts:
        output["sample_prompts"].append(prompt.prompt)

    status.job_no = 3
    status.textinfo = "Building dataset for 'new' class images..."
    for concept in config.concepts_list:
        c_idx = 0
        class_images_dir = Path(concept["class_data_dir"])
        if class_images_dir == "" or class_images_dir is None or class_images_dir == db_shared.script_path:
            class_images_dir = os.path.join(config.model_dir, f"classifiers_{c_idx}")
        class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = len(get_images(class_images_dir))
        if cur_class_images < concept.num_class_images:
            sample_dataset = PromptDataset(config.concepts_list, config.model_dir, config.resolution)
            for i in mytqdm(range(sample_dataset.__len__()), desc="Generating prompts"):
                prompt = sample_dataset.__getitem__(i)
                output["new_class_prompts"].append(prompt.prompt)
        c_idx += 1
    status.job_no = 4
    status.textinfo = "Prompt generation complete."
    return json.dumps(output)


def generate_dataset(model_name: str, instance_prompts: List[PromptData] = None, class_prompts: List[PromptData] = None,
                     batch_size = None, tokenizer=None, vae=None, debug=True):
    if debug:
        print("Generating dataset.")
    db_gallery = gradio.update(value=None)
    db_prompt_list = gradio.update(value=None)
    db_status = gradio.update(value=None)

    args = from_file(model_name)

    if batch_size is None:
        batch_size = args.train_batch_size

    if args is None:
        print("No CONFIG!")
        return db_gallery, db_prompt_list, db_status

    tokens = []

    print(f"Found {len(class_prompts)} reg images.")

    min_bucket_reso = (int(args.resolution * 0.28125) // 64) * 64
    from extensions.sd_dreambooth_extension.dreambooth.finetuning_dataset import DbDataset

    print("Preparing dataset...")
    train_dataset = DbDataset(
        batch_size=batch_size,
        instance_prompts=instance_prompts,
        class_prompts=class_prompts,
        tokens=tokens,
        tokenizer=tokenizer,
        resolution=args.resolution,
        prior_loss_weight=args.prior_loss_weight,
        hflip=args.hflip,
        random_crop=args.center_crop,
        shuffle_tokens=args.shuffle_tags,
        not_pad_tokens=not args.pad_tokens,
        debug_dataset=debug
    )
    train_dataset.make_buckets_with_caching(vae, min_bucket_reso)
    #train_dataset = train_dataset.pin_memory()
    print(f"Total dataset length (steps): {len(train_dataset)}")
    return train_dataset


def generate_classifiers(args: DreamboothConfig, use_txt2img: bool = True, accelerator: Accelerator = None, ui=False):
    """

    @param args: A DreamboothConfig
    @param use_txt2img: Generate images using txt2image. Does not use lora.
    @param accelerator: An optional existing accelerator to use.
    @param ui: Whether this was called by th UI, or is being run during training.
    @return:
    generated: Number of images generated
    images: A list of images or image paths, depending on if returning to the UI or not.
    if ui is False, this will return a second array of paths representing the class paths.
    """
    out_images = []
    instance_prompts = []
    class_prompts = []
    try:
        status.textinfo = "Preparing dataset..."
        prompt_dataset = PromptDataset(args.concepts_list, args.model_dir, args.resolution)
        instance_prompts = prompt_dataset.instance_prompts
        class_prompts = prompt_dataset.class_prompts
    except Exception as p:
        print(f"Exception generating dataset: {str(p)}")
        traceback.print_exc()
        if ui:
            db_shared.status.end()
            return 0, []
        else:
            return 0, instance_prompts, class_prompts

    set_len = prompt_dataset.__len__()
    if set_len == 0:
        print("Nothing to generate.")
        if ui:
            db_shared.status.end()
            return 0, []
        else:
            return 0, instance_prompts, class_prompts

    print(f"Generating {set_len} class images for training...")
    status.textinfo = f"Generating {set_len} class images for training..."
    status.job_count = set_len
    status.job_no = 0
    builder = ImageBuilder(args, use_txt2img=use_txt2img, lora_model=args.lora_model_name, lora_weight=args.lora_weight,
                           lora_txt_weight=args.lora_txt_weight, batch_size=args.sample_batch_size, accelerator=accelerator)
    generated = 0
    actual_idx = 0
    pbar = mytqdm(total=set_len, desc="Generating class images")
    for i in range(set_len):
        first_res = None
        if status.interrupted or generated >= set_len:
            break
        prompts = []
        # Decrease batch size
        if set_len - generated < args.sample_batch_size:
            batch_size = set_len - generated
        else:
            batch_size = args.sample_batch_size
        for b in range(batch_size):
            pd = prompt_dataset.__getitem__(actual_idx)
            # Ensure that our image batches have the right resolutions
            if first_res is None:
                first_res = pd.resolution
            if pd.resolution == first_res:
                prompts.append(pd)
                actual_idx += 1

        new_images = builder.generate_images(prompts)
        i_idx = 0
        preview_images = []
        preview_prompts = []
        for image in new_images:
            if generated >= set_len:
                break
            try:
                pd = prompts[i_idx]
                image_filename = db_save_image(image, pd)
                class_prompts.append(pd)
                if ui:
                    out_images.append(image)
                pbar.update()
                i_idx += 1
                generated += 1
                preview_images.append(image_filename)
                preview_prompts.append(pd.prompt)
                status.textinfo = f"Class image(s) {generated}/{set_len}:'"
            except Exception as e:
                print(f"Exception generating images: {e}")
                traceback.print_exc()

        status.current_image = preview_images
        status.sample_prompts = preview_prompts
    builder.unload(ui)
    del prompt_dataset
    cleanup()
    print(f"Generated {generated} new class images.")
    if ui:
        db_shared.status.end()
        return generated, out_images
    else:
        return generated, instance_prompts, class_prompts


# Implementation from https://github.com/bmaltais/kohya_ss
def encode_hidden_state(text_encoder: CLIPTextModel, input_ids, pad_tokens, b_size, max_token_length,
                        tokenizer_max_length):
    if pad_tokens:
        input_ids = input_ids.reshape((-1, tokenizer_max_length))  # batch_size*3, 77

    clip_skip = db_shared.CLIP_stop_at_last_layers
    if clip_skip <= 1:
        encoder_hidden_states = text_encoder(input_ids)[0]
    else:
        enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out['hidden_states'][-clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    if not pad_tokens:
        return encoder_hidden_states

    encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

    if max_token_length > 75:
        sts_list = [encoder_hidden_states[:, 0].unsqueeze(1)]
        for i in range(1, max_token_length, tokenizer_max_length):
            sts_list.append(encoder_hidden_states[:, i:i + tokenizer_max_length - 2])
        sts_list.append(encoder_hidden_states[:, -1].unsqueeze(1))
        encoder_hidden_states = torch.cat(sts_list, dim=1)

    return encoder_hidden_states


class TrainResult:
    config: DreamboothConfig = None
    msg: str = ""
    samples: [Image] = []
