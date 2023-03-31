from __future__ import annotations

import hashlib
import math
import os
import random
import re
import sys
from io import StringIO

from diffusers.schedulers import KarrasDiffusionSchedulers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import features, PngImagePlugin, Image, ExifTags

import os
from typing import List, Tuple, Dict, Union

import numpy as np
import torch
import torch.utils.checkpoint

from dreambooth.dataclasses.db_concept import Concept
from dreambooth.dataclasses.prompt_data import PromptData
from helpers.mytqdm import mytqdm
from dreambooth import shared
from dreambooth.shared import status


def get_dim(filename, max_res):
    with Image.open(filename) as im:
        im = rotate_image_straight(im)
        width, height = im.size
        if width > max_res or height > max_res:
            aspect_ratio = width / height
            if width > height:
                width = max_res
                height = int(max_res / aspect_ratio)
            else:
                height = max_res
                width = int(max_res * aspect_ratio)
        return width, height


def rotate_image_straight(image: Image) -> Image:
    exif: Image.Exif = image.getexif()
    if exif:
        orientation_tag = {v: k for k, v in ExifTags.TAGS.items()}['Orientation']
        orientation = exif.get(orientation_tag)
        degree = {
            3: 180,
            6: 270,
            8: 90,
        }.get(orientation)
        if degree:
            image = image.rotate(degree, expand=True)
    # else:
    #     print(f"No exif data for {image.filename}. Using default orientation.")
    return image


def get_images(image_path: str):
    pil_features = list_features()
    output = []
    if os.path.exists(image_path):
        for file in os.listdir(image_path):
            # Mac creates metadata files for every image with name `._{filename}`, so we skip it
            if sys.platform == 'darwin' and file.startswith('._'):
                continue
            file_path = os.path.join(image_path, file)
            if is_image(file_path, pil_features):
                output.append(file_path)
            if os.path.isdir(file_path):
                sub_images = get_images(file_path)
                for image in sub_images:
                    output.append(image)
    return output


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


def is_image(path: str, feats=None):
    if feats is None:
        feats = []
    if not len(feats):
        feats = list_features()
    is_img = os.path.isfile(path) and os.path.splitext(path)[1].lower() in feats
    return is_img


def sort_prompts(
        concept: Concept,
        text_getter: FilenameTextGetter,
        img_dir: str,
        images: List[str],
        bucket_resos: List[Tuple[int, int]],
        concept_index: int,
        is_class: bool,
        pbar: mytqdm,
        verbatim=False
) -> Dict[Tuple[int, int], PromptData]:
    prompts = {}
    max_dim = 0
    for (w, h) in bucket_resos:
        if w > max_dim:
            max_dim = w
        if h > max_dim:
            max_dim = h
    _, dirr = os.path.split(img_dir)
    for img in images:
        # Get prompt
        pbar.set_description(f"Pre-processing images: {dirr}")

        file_text = text_getter.read_text(img)
        if verbatim:
            prompt = file_text
        else:
            prompt = text_getter.create_text(
                concept.class_prompt if is_class else concept.instance_prompt,
                file_text,
                concept,
                is_class
            )

        w, h = get_dim(img, max_dim)
        reso = closest_resolution(w, h, bucket_resos)
        prompt_list = prompts[reso] if reso in prompts else []
        pd = PromptData(
            prompt=prompt,
            negative_prompt=concept.class_negative_prompt if is_class else None,
            instance_token=concept.instance_token,
            class_token=concept.class_token,
            src_image=img,
            resolution=reso,
            concept_index=concept_index
        )
        prompt_list.append(pd)
        pbar.update()
        prompts[reso] = prompt_list
    return dict(sorted(prompts.items()))


class FilenameTextGetter:
    """Adapted from modules.textual_inversion.dataset.PersonalizedBase to get caption for image."""

    re_numbers_at_start = re.compile(r"^[-\d]+\s*")

    def __init__(self, shuffle_tags=False):
        self.re_word = re.compile(shared.dataset_filename_word_regex) if len(
            shared.dataset_filename_word_regex) > 0 else None
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
                filename_text = (shared.dataset_filename_join_string or "").join(tokens)

        return filename_text

    def create_text(self, prompt, file_text, concept, is_class=True):
        instance_token = concept.instance_token
        class_token = concept.class_token
        output = prompt.replace("[filewords]", file_text)

        if instance_token and class_token:
            instance_regex = re.compile(f"\\b{instance_token}\\b", flags=re.IGNORECASE)
            class_regex = re.compile(f"\\b{class_token}\\b", flags=re.IGNORECASE)
            extended_class_regexes = list(
                re.compile(r) for r in [f"a {class_token}", f"the {class_token}", f"an {class_token}", class_token])

            if is_class:
                if instance_regex.search(output):
                    if class_regex.search(output):
                        output = instance_regex.sub("", output)
                    else:
                        output = instance_regex.sub(class_token, output)
                if not class_regex.search(output):
                    output = f"{class_token}, {output}"

            else:
                if class_regex.search(output):
                    # Do nothing if we already have class and instance in string
                    if instance_regex.search(output):
                        pass
                    # Otherwise, substitute class tokens for the base token
                    else:
                        for extended_class_regex in extended_class_regexes:
                            output = extended_class_regex.sub(class_token, output)

                        # Now, replace class with instance + class tokens
                        output = class_regex.sub(f"{instance_token}", output)
                else:
                    # If class is not in the string, check if instance is
                    if instance_regex.search(output):
                        output = instance_regex.sub(f"{instance_token}", output)
                    else:
                        # Description only, insert both at the front?
                        output = f"{instance_token}, {output}"

        elif instance_token and not is_class:
            output = f"{instance_token}, {output}"

        elif class_token and is_class:
            output = f"{class_token}, {output}"

        output = re.sub(r"\s+", " ", output)
        output = re.sub(r"\\", "", output)

        if self.shuffle_tags:
            output = shuffle_tags(output)
        else:
            output = output.strip()

        return output


def shuffle_tags(caption: str):
    tags = caption.split(',')
    first_tag = tags.pop(0)
    random.shuffle(tags)
    tags.insert(0, first_tag)
    output = ','.join(tags).strip()
    return output


def get_scheduler_names():
    return [scheduler.name.replace('Scheduler', '') for scheduler in KarrasDiffusionSchedulers]


def get_scheduler_class(scheduler_name):
    try:
        # Get the class type by name from the KarrasDiffusionSchedulers enum
        scheduler_class = getattr(sys.modules["diffusers"], scheduler_name + 'Scheduler')
    except AttributeError:
        raise ValueError(f"No scheduler named {scheduler_name} found")

    return scheduler_class


def make_bucket_resolutions(max_resolution, divisible=8) -> List[Tuple[int, int]]:
    aspect_ratios = [(16, 9), (5, 4), (4, 3), (3, 2), (2, 1), (1, 1)]
    resos = set()

    for ar in aspect_ratios:
        w = int(max_resolution * math.sqrt(ar[0] / ar[1]) // divisible) * divisible
        h = int(max_resolution * math.sqrt(ar[1] / ar[0]) // divisible) * divisible

        resos.add((w, h))
        resos.add((h, w))

    resos = list(resos)
    resos.sort()
    return resos


def closest_resolution(img_w, img_h, resos) -> Tuple[int, int]:
    img_ratio = img_w / img_h

    def distance(res):
        res_w, res_h = res
        res_ratio = res_w / res_h
        return abs(img_ratio - res_ratio)

    return min(resos, key=distance)


txt2img_available = False
try:
    from modules import devices, sd_hijack, prompt_parser, lowvram
    from modules.processing import StableDiffusionProcessing, Processed, \
        get_fixed_seed, create_infotext, decode_first_stage
    from modules.sd_hijack import model_hijack

    txt2img_available = True


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
            p.all_prompts = p.prompt
        else:
            p.all_prompts = p.batch_size * p.n_iter * [p.prompt]

        if type(p.negative_prompt) == list:
            p.all_negative_prompts = p.negative_prompt
        else:
            p.all_negative_prompts = p.batch_size * p.n_iter * [p.negative_prompt]

        if type(seed) == list:
            p.all_seeds = seed
        else:
            p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

        if type(subseed) == list:
            p.all_subseeds = subseed
        else:
            p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

        def infotext(iteration=0, position_in_batch=0):
            return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration,
                                   position_in_batch)

        with open(os.path.join(shared.script_path, "params.txt"), "w", encoding="utf8") as file:
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
                    samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds,
                                            subseeds=subseeds,
                                            subseed_strength=p.subseed_strength, prompts=prompts)

                x_samples_ddim = [
                    decode_first_stage(p.sd_model, samples_ddim[i:i + 1].to(dtype=devices.dtype_vae))[0].cpu()
                    for i in range(samples_ddim.size(0))]
                x_samples_ddim = torch.stack(x_samples_ddim).float()
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                del samples_ddim

                if shared.lowvram or shared.medvram:
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
except:
    print("Oops, no txt2img available. Oh well.")


    def process_txt2img(p: StableDiffusionProcessing) -> None:
        return None


def load_image_directory(db_dir, concept: Concept, is_class: bool = True) -> List[Tuple[str, str]]:
    img_paths = get_images(db_dir)
    captions = []
    text_getter = FilenameTextGetter()
    for img_path in img_paths:
        file_text = text_getter.read_text(img_path)
        final_caption = text_getter.create_text(
            concept.instance_prompt,
            file_text,
            concept,
            is_class
        )
        captions.append(final_caption)

    return list(zip(img_paths, captions))


def open_and_trim(image_path: str, reso: Tuple[int, int], return_pil: bool = False) -> Union[np.ndarray, Image]:
    # Open image with PIL
    image = Image.open(image_path)
    image = rotate_image_straight(image)

    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Upscale image if necessary
    scale_factor = max(reso[0] / image.width, reso[1] / image.height)
    if scale_factor != 1:
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        image = image.resize(new_size, resample=Image.LANCZOS)

    # Crop image to target resolution
    if image.width != reso[0] or image.height != reso[1]:
        w = int((image.width - reso[0]) / 2)
        h = int((image.height - reso[1]) / 2)
        box = (w, h, reso[0] + w, reso[1] + h)
        image = image.crop(box)

    # Return as np array or PIL image
    if return_pil:
        return image
    else:
        return np.array(image)


def db_save_image(image: Image, prompt_data: PromptData = None, save_txt: bool = True, custom_name: str = None):
    image_base = hashlib.sha1(image.tobytes()).hexdigest()

    file_name = image_base
    if custom_name is not None:
        file_name = custom_name

    file_name = re.sub(r"[^\w \-_.]", "", file_name)

    image_filename = os.path.join(prompt_data.out_dir, f"{file_name}.tmp")
    pnginfo_data = PngImagePlugin.PngInfo()
    if prompt_data is not None:
        size = prompt_data.resolution
        generation_params = {
            "Steps": prompt_data.steps,
            "CFG scale": prompt_data.scale,
            "Seed": prompt_data.seed,
            "Size": f"{size[0]}x{size[1]}"
        }

        generation_params_text = ", ".join(
            [k if k == v else f'{k}: {f"{v}" if "," in str(v) else v}' for k, v in generation_params.items()
             if v is not None])

        prompt_string = f"{prompt_data.prompt}\nNegative prompt: {prompt_data.negative_prompt}\n{generation_params_text}".strip()
        pnginfo_data.add_text("parameters", prompt_string)

    image_format = Image.registered_extensions()[".png"]

    image.save(image_filename, format=image_format, pnginfo=pnginfo_data)

    if save_txt and prompt_data is not None:
        os.replace(image_filename, image_filename)
        txt_filename = image_filename.replace(".tmp", ".txt")
        with open(txt_filename, "w", encoding="utf8") as file:
            file.write(prompt_data.prompt)
    os.replace(image_filename, image_filename.replace(".tmp", ".png"))
    return image_filename.replace(".tmp", ".png")


def image_grid(imgs):
    rows = math.floor(math.sqrt(len(imgs)))
    while len(imgs) % rows != 0:
        rows -= 1

    if rows > len(imgs):
        rows = len(imgs)

    cols = math.ceil(len(imgs) / rows)

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color='black')

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid
