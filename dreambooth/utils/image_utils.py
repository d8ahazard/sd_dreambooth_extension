from __future__ import annotations

import hashlib
import math
import os
import random
import re
from io import StringIO


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import features, PngImagePlugin, Image

import os
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.utils.checkpoint

try:
    from extensions.sd_dreambooth_extension.dreambooth.dataclasses.db_concept import Concept
    from extensions.sd_dreambooth_extension.dreambooth.dataclasses.prompt_data import PromptData
    from extensions.sd_dreambooth_extension.helpers.mytqdm import mytqdm
    
    from extensions.sd_dreambooth_extension.dreambooth import shared
    from extensions.sd_dreambooth_extension.dreambooth.shared import status
except:
    from dreambooth.dataclasses.db_concept import Concept # noqa
    from dreambooth.dataclasses.prompt_data import PromptData # noqa
    from helpers.mytqdm import mytqdm # noqa

    from dreambooth import shared # noqa
    from dreambooth.shared import status # noqa



def get_dim(filename, max_res):
    with Image.open(filename) as im:
        width, height = im.size
        try:
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
        except:
            print(f"No exif data for {filename}. Using default orientation.")
        return width, height


def get_images(image_path:str):
    pil_features = list_features()
    output = []
    if os.path.exists(image_path):
        for file in os.listdir(image_path):
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

def sort_prompts(concept: Concept, text_getter: FilenameTextGetter, img_dir: str, images: List[str], bucket_resos: List[Tuple[int, int]],
                 concept_index: int, is_class: bool, pbar: mytqdm) -> Dict[Tuple[int, int], PromptData]:
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
        text = text_getter.read_text(img)
        prompt = text_getter.create_text(
            concept.class_prompt if is_class else concept.instance_prompt,
            text, concept.instance_token, concept.class_token, is_class)
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

        filename_text = filename_text.replace("\\", "")  # work with \(franchies\)
        return filename_text

    def create_text(self, text_template, filename_text, instance_token, class_token, is_class=True):
        # Append the filename text to the template first...THEN shuffle it all.
        filename_text = text_template.replace("[filewords]", filename_text)
        # If we are creating text for a class image and it has our instance token in it, remove/replace it
        class_tokens = [f"a {class_token}", f"the {class_token}", f"an {class_token}", class_token]
        if instance_token != "" and class_token != "":
            if is_class and re.search(f'(^|\\W){instance_token}($|\\W)', filename_text):
                if re.search(f'(^|\\W){class_token}($|\\W)', filename_text):
                    filename_text = filename_text.replace(instance_token, "")
                    filename_text = filename_text.replace("  ", " ")
                else:
                    filename_text = filename_text.replace(instance_token, class_token)

            if not is_class:
                if re.search(f'(^|\\W){class_token}($|\\W)', filename_text):
                    # Do nothing if we already have class and instance in string
                    if re.search(f'(^|\\W){instance_token}($|\\W)', filename_text):
                        pass
                    # Otherwise, substitute class tokens for the base token
                    else:
                        for token in class_tokens:
                            if re.search(f'(^|\\W){token}($|\\W)', filename_text):
                                filename_text = filename_text.replace(token, f"{class_token}")
                    # Now, replace class with instance + class tokens
                    filename_text = filename_text.replace(class_token, f"{instance_token} {class_token}")
                else:
                    # If class is not in the string, check if instance is
                    if re.search(f'(^|\\W){instance_token}($|\\W)', filename_text):
                        filename_text = filename_text.replace(instance_token, f"{instance_token} {class_token}")
                    else:
                        # Description only, insert both at the front?
                        filename_text = f"{instance_token} {class_token}, {filename_text}"

        # We already replaced [filewords] up there ^^
        output = filename_text
        # Remove underscores, double-spaces, and other characters that will cause issues.
        output = output.replace("_", " ")
        output = output.replace("  ", " ")
        strip_chars = ["(", ")", "/", "\\", ":", "[", "]"]
        for s_char in strip_chars:
            output = output.replace(s_char, "")

        tags = output.split(',')

        if self.shuffle_tags and len(tags) > 2:
            first_tag = tags.pop(0)
            random.shuffle(tags)
            tags.insert(0, first_tag)

        output = ','.join(tags)
        return output.strip()


def make_bucket_resolutions(max_size, min_size=256, divisible=64) -> List[Tuple[int, int]]:
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


def closest_resolution(width, height, resos) -> Tuple[int, int]:
    def distance(reso):
        w, h = reso
        if w > width or h > height:
            return float("inf")
        return (w - width) ** 2 + (h - height) ** 2

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
            return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

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
                    samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds,
                                            subseed_strength=p.subseed_strength, prompts=prompts)

                x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i + 1].to(dtype=devices.dtype_vae))[0].cpu()
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

def load_image_directory(db_dir, concept: Concept, is_class: bool = True) -> List[Tuple[str, str]]:
    img_paths = get_images(db_dir)
    captions = []
    text_getter = FilenameTextGetter()
    for img_path in img_paths:
        cap_for_img = text_getter.read_text(img_path)
        final_caption = text_getter.create_text(concept.instance_prompt, cap_for_img, concept.instance_token,
                                                concept.class_token, is_class)
        captions.append(final_caption)

    return list(zip(img_paths, captions))


def db_save_image(image: Image, prompt_data: PromptData = None, save_txt: bool = True, custom_name: str = None):
    image_base = hashlib.sha1(image.tobytes()).hexdigest()
    image_filename = os.path.join(prompt_data.out_dir, f"{image_base}.tmp")
    if custom_name is not None:
        image_filename = os.path.join(prompt_data.out_dir, f"{custom_name}.tmp")

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
