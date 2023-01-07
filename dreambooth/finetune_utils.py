import gc
import hashlib
import json
import os
import random
import re
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

import gradio
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from diffusers import DiffusionPipeline, AutoencoderKL
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import CLIPTextModel, AutoTokenizer

from extensions.sd_dreambooth_extension.dreambooth import db_shared
from extensions.sd_dreambooth_extension.dreambooth.db_concept import Concept
from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig, from_file
from extensions.sd_dreambooth_extension.dreambooth.db_shared import status
from extensions.sd_dreambooth_extension.dreambooth.utils import cleanup, get_checkpoint_match, get_images
from extensions.sd_dreambooth_extension.lora_diffusion.lora import apply_lora_weights
from modules import shared, devices, sd_models, sd_hijack, prompt_parser, lowvram
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessing, Processed, \
    get_fixed_seed, create_infotext, decode_first_stage
from modules.sd_hijack import model_hijack


@dataclass
class PromptData:
    prompt:str = ""
    negative_prompt:str = ""
    steps:int = 60
    scale:float = 7.5
    out_dir:str = ""
    seed:int = -1
    resolution: Tuple[int, int] = (512, 512)

    @property
    def __dict__(self):
        """
        get a python dictionary
        """
        return asdict(self)

    @property
    def json(self):
        """
        get the json formated string
        """
        return json.dumps(self.__dict__)


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
        output = ','.join(tags)
        return output


from PIL import Image

def get_dim(filename):
    with Image.open(filename) as im:
        width, height = im.size
        exif = im.getexif()
        if exif:
            orientation = exif.get(274)
            if orientation == 3 or orientation == 6:
                width, height = height, width
        return width, height


def sort_prompts(concept: Concept, text_getter: FilenameTextGetter, img_dir: str, bucket_resos, instance_prompts = None) -> Dict[Tuple[int, int], List[str]]:
    prompts = {}
    images = get_images(img_dir)

    for img in images:
        # Get prompt
        text = text_getter.read_text(img)
        prompt = text_getter.create_text(concept.class_prompt, text, concept.instance_token, concept.class_token)
        w, h = get_dim(img)
        reso = closest_resolution(w, h, bucket_resos)
        if instance_prompts is not None:
            if reso not in instance_prompts:
                continue
            matched = False
            for instance_prompt in instance_prompts[reso]:
                if compare_prompts(instance_prompt[0], prompt,[(concept.instance_token, concept.class_token)]):
                    matched = True
                    break
            if not matched:
                continue
        prompt_list = prompts[reso] if reso in prompts else []
        prompt_list.append((prompt, (w, h), img))
        prompts[reso] = prompt_list
            
    return dict(sorted(prompts.items()))

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

class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, concepts: [Concept], model_dir: str, max_width:int):
        min_width = (int(max_width * 0.28125) // 64) * 64
        c_idx = 0
        text_getter = FilenameTextGetter(False)
        self.total_len = 0
        # Loop concepts
        prompts_to_create = {}
        num_prompts_to_create = 0
        bucket_resos, bucket_aspect_ratios = make_bucket_resolutions(max_width, min_width)
        self.instance_paths = []
        self.class_paths = []
        for concept in concepts:
            instance_dir = concept.instance_data_dir
            class_dir = concept.class_data_dir
            
            # Filter empty class dir and set if necessary
            if class_dir == "" or class_dir is None or class_dir == db_shared.script_path:
                class_dir = os.path.join(model_dir, f"classifiers_{c_idx}")
            class_dir = Path(class_dir)
            class_dir.mkdir(parents=True, exist_ok=True)

            status.textinfo = "Sorting images..."
            # Sort existing prompts
            instance_prompts = sort_prompts(concept, text_getter, instance_dir, bucket_resos)
            if concept.num_class_images_per >= 0:
                class_prompts = sort_prompts(concept, text_getter, class_dir, bucket_resos, instance_prompts)
            else:
                class_prompts = {}
            idx = 0
            matched_resos = []
            for res, prompts in instance_prompts.items():
                for prompt in prompts:
                    self.instance_paths.append(prompt[2])
                print(f"Instance Bucket {idx}: Resolution {res}, Count: {len(prompts)}")
                if len(prompts) > 0:
                    matched_resos.append((idx, res))
                idx += 1

            if concept.num_class_images_per > 0:
                for idx, res in matched_resos:
                    prompts = []
                    if res in class_prompts:
                        prompts = class_prompts[res]
                        for prompt in prompts:
                            self.class_paths.append(prompt[2])
                    print(f"Class Bucket {idx}: Resolution {res}, Count: {len(prompts)}")

            # Loop by resolutions
            for res, inst_prompts in instance_prompts.items():
                cls_prompts = class_prompts[res] if res in class_prompts else []
                new_prompts = prompts_to_create[res] if res in prompts_to_create else []
                if "[filewords]" not in concept.class_prompt:
                    num_res_img = len(inst_prompts) * concept.num_class_images_per
                    if len(cls_prompts) < num_res_img:
                        num_prompts = num_res_img - len(cls_prompts)
                        prompt_data = PromptData(
                            concept.class_prompt,
                            concept.class_negative_prompt,
                            concept.class_infer_steps,
                            concept.class_guidance_scale,
                            class_dir,
                            -1,
                            res
                        )
                        new_prompts = [prompt_data] * num_prompts
                        num_prompts_to_create += num_prompts
                else:
                    for prompt, actual_res, path in inst_prompts:
                        num_matches = 0
                        for cl_cap, cl_res, cl_path in cls_prompts:
                            if compare_prompts(prompt, cl_cap, [(concept.instance_token, concept.class_token)]):
                                num_matches += 1
                        if num_matches < concept.num_class_images_per:
                            num_prompts = concept.num_class_images_per - num_matches
                            pd = PromptData(
                                    prompt,
                                    concept.class_negative_prompt,
                                    concept.class_infer_steps,
                                    concept.class_guidance_scale,
                                    class_dir,
                                    -1,
                                    res
                                )
                            prompts = [
                                pd
                            ] * num_prompts
                            new_prompts.extend(prompts)
                            num_prompts_to_create += num_prompts
                prompts_to_create[res] = new_prompts
            c_idx += 1
        new_len = 0
        idx = 0
        for w, h in prompts_to_create:
            prompts = prompts_to_create[(w, h)]
            new_len += len(prompts)
            print(f"Target Bucket {idx}: Resolution {w, h}, Count: {len(prompts)}")
            idx += 1
        self.total_len = new_len
        print(f"We need a total of {new_len} images.")
        self.prompts = prompts_to_create

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, index) -> PromptData:
        res_index = 0
        for res, prompt_datas in self.prompts.items():
            for p in range(len(prompt_datas)):
                if res_index == index:
                    return prompt_datas[p]
                res_index += 1
        print(f"Invalid index: {index}/{self.total_len}")
        return None


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


def make_bucket_resolutions(max_size, min_size=256, divisible=64):
    resos = set()

    for w in range(min_size, max_size + divisible, divisible):
        for h in range(min_size, max_size + divisible, divisible):
            resos.add((w, h))
            resos.add((h, w))

    resos = list(resos)
    resos.sort()

    aspect_ratios = [w / h for w, h in resos]
    return resos, aspect_ratios

def closest_resolution(width, height, resos):
    def distance(reso):
        w, h = reso
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

class ImageBuilder:
    def __init__(self, config: DreamboothConfig, use_txt2img: bool, lora_model: str = None, lora_weight: float = 1,
                 lora_txt_weight: float = 1, batch_size: int = 1, accelerator: Accelerator = None):
        self.image_pipe = None
        self.txt_pipe = None
        self.resolution = config.resolution
        self.last_model = None
        self.batch_size = batch_size
        if not os.path.exists(config.src) and use_txt2img:
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
                apply_lora_weights(lora_model, self.image_pipe.unet, self.image_pipe.text_encoder, lora_weight,
                                   lora_txt_weight,
                                   accelerator.device)
        else:
            current_model = sd_models.select_checkpoint()
            new_model_info = get_checkpoint_match(config.src)
            if new_model_info is not None and current_model is not None:
                if new_model_info[0] != current_model[0]:
                    self.last_model = current_model
                    print(f"Loading model: {new_model_info[0]}")
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
            for i in range(sample_dataset.__len__()):
                prompt = sample_dataset.__getitem__(i)
                output["new_class_prompts"].append(prompt.prompt)
        c_idx += 1
    status.job_no = 4
    status.textinfo = "Prompt generation complete."
    return json.dumps(output)


def generate_dataset(model_name: str, instance_paths = None, class_paths = None, batch_size = None, tokenizer=None, vae=None, debug=True):
    db_gallery = gradio.update(value=None)
    db_prompt_list = gradio.update(value=None)
    db_status = gradio.update(value=None)

    args = from_file(model_name)

    # If debugging, we pass no batch size and set to 1, so we can enumerate all items
    if batch_size is None:
        batch_size = 1

    if args is None:
        print("No CONFIG!")
        return db_gallery, db_prompt_list, db_status

    train_img_path_captions = []
    reg_img_path_captions = []
    tokens = []
    use_concepts = False
    for conc in args.concepts_list:
        if not conc.is_valid():
            continue
        if conc.num_class_images_per > 0:
            use_concepts = True
        if instance_paths is not None and class_paths is not None:
            train_img_path_captions.extend(get_captions(conc, instance_paths, False))
            if conc.num_class_images_per > 0:
                reg_img_path_captions.extend(get_captions(conc, class_paths, True))
        else:
            if conc.class_token != "" and conc.instance_token != "":
                tokens.append((conc.instance_token, conc.class_token))
            idd = conc.instance_data_dir
            if idd is not None and idd != "" and os.path.exists(idd):
                img_caps = load_dreambooth_dir(idd, conc, False)
                if conc.num_class_images_per > 0:
                    train_img_path_captions.extend(img_caps)
                    print(f"Found {len(train_img_path_captions)} training images.")

            class_data_dir = conc.class_data_dir
            number_class_images = conc.num_class_images_per
            if number_class_images > 0 and class_data_dir is not None and class_data_dir != "" and os.path.exists(
                    class_data_dir):
                reg_caps = load_dreambooth_dir(class_data_dir, conc)
                reg_img_path_captions.extend(reg_caps)
    if use_concepts:
        print(f"Found {len(reg_img_path_captions)} reg images.")

    min_bucket_reso = (int(args.resolution * 0.28125) // 64) * 64
    from extensions.sd_dreambooth_extension.dreambooth.finetuning_dataset import DreamBoothOrFineTuningDataset

    print("Preparing dataset")
    train_dataset = DreamBoothOrFineTuningDataset(
        batch_size=batch_size,
        train_img_path_captions=train_img_path_captions,
        reg_img_path_captions=reg_img_path_captions,
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
    print(f"Total dataset length (steps): {len(train_dataset)}")
    if debug:
        images = []
        prompts = []
        for example in train_dataset:
            ex_images = example["images"]
            ex_prompts = example["captions"]
            if len(ex_images) == 1:
                res = ex_images[0].size
                b_res = example["res"]
                cap = ex_prompts[0]
                images.append(ex_images[0])
                prompts.append(f"{res} B{b_res}: {cap}")
        if len(prompts) > 0:
            message = "Images Missing Classes:<br>"
            for p in prompts:
                message += f"{p}<br>"
        else:
            message = "No missing class images."
            status.textinfo = message
        return images, prompts, message
    else:
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
    inst_paths = []
    class_paths = []
    try:
        status.textinfo = "Preparing dataset..."
        prompt_dataset = PromptDataset(args.concepts_list, args.model_dir, args.resolution)
        inst_paths = prompt_dataset.instance_paths
        class_paths = prompt_dataset.class_paths
    except Exception as p:
        print(f"Exception generating dataset: {str(p)}")
        traceback.print_exc()
        if ui:
            return 0, []
        else:
            return 0, inst_paths, class_paths

    set_len = prompt_dataset.__len__()
    if set_len == 0:
        print("Nothing to generate.")
        if ui:
            return 0, []
        else:
            return 0, inst_paths, class_paths

    print(f"Generating {set_len} class images for training...")
    status.textinfo = f"Generating {set_len} class images for training..."
    status.job_count = set_len
    status.job_no = 0
    builder = ImageBuilder(args, use_txt2img=use_txt2img, lora_model=args.lora_model_name, lora_weight=args.lora_weight,
                           lora_txt_weight=args.lora_txt_weight, batch_size=args.sample_batch_size, accelerator=accelerator)
    generated = 0
    pbar = tqdm(total=set_len)
    actual_idx = 0
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
                print(f"Generating: {pd}")
                prompts.append(pd)
                actual_idx += 1

        new_images = builder.generate_images(prompts)
        i_idx = 0
        preview_images = []
        for image in new_images:
            if generated >= set_len:
                break
            try:
                pd = prompts[i_idx]
                image_base = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = os.path.join(pd.out_dir, f"{image_base}.png")
                image.save(image_filename)
                class_paths.append(image_filename)
                if ui:
                    out_images.append(image)
                else:
                    class_paths.append(image_filename)
                txt_filename = image_filename.replace(".png", ".txt")
                with open(txt_filename, "w", encoding="utf8") as file:
                    file.write(pd.prompt)
                status.job_no += 1
                i_idx += 1
                generated += 1
                preview_images.append(image)
                status.textinfo = f"Class image {generated}/{set_len}, Prompt: '{pd.prompt}'"
                if pbar is not None:
                    pbar.update()
            except Exception as e:
                print(f"Exception generating images: {e}")
                traceback.print_exc()

        status.current_image = preview_images
    builder.unload(ui)
    del prompt_dataset
    cleanup()
    print(f"Generated {generated} new class images.")
    if ui:
        return generated, out_images
    else:
        print("UI RETURN.")
        return generated, inst_paths, class_paths


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
