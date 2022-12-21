import gc
import hashlib
import json
import os
import random
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from diffusers import DiffusionPipeline, AutoencoderKL
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import CLIPTextModel, AutoTokenizer

from extensions.sd_dreambooth_extension.dreambooth import dream_state
from extensions.sd_dreambooth_extension.dreambooth.db_concept import Concept
from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig, from_file
from extensions.sd_dreambooth_extension.dreambooth.dream_state import status
from extensions.sd_dreambooth_extension.dreambooth.utils import printm, cleanup, get_checkpoint_match, get_images
from extensions.sd_dreambooth_extension.lora_diffusion.lora import apply_lora_weights
from modules import shared, devices, sd_models, images, sd_hijack, prompt_parser, lowvram
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessing, Processed, \
    get_fixed_seed, create_infotext, decode_first_stage
from modules.sd_hijack import model_hijack


@dataclass
class PromptData:
    prompt = ""
    negative_prompt = ""
    steps = 60
    scale = 7.5
    out_dir = ""
    seed = -1


class FilenameTextGetter:
    """Adapted from modules.textual_inversion.dataset.PersonalizedBase to get caption for image."""

    re_numbers_at_start = re.compile(r"^[-\d]+\s*")

    def __init__(self, shuffle_tags=False):
        self.re_word = re.compile(shared.opts.dataset_filename_word_regex) if len(
            shared.opts.dataset_filename_word_regex) > 0 else None
        self.shuffle_tags = shuffle_tags

    def read_text(self, img_path):
        text_filename = os.path.splitext(img_path)[0] + ".txt"
        filename = os.path.basename(img_path)

        if os.path.exists(text_filename):
            with open(text_filename, "r", encoding="utf8") as file:
                filename_text = file.read()
        else:
            filename_text = os.path.splitext(filename)[0]
            filename_text = re.sub(self.re_numbers_at_start, '', filename_text)
            if self.re_word:
                tokens = self.re_word.findall(filename_text)
                filename_text = (shared.opts.dataset_filename_join_string or "").join(tokens)

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

        tags = filename_text.split(',')
        if self.shuffle_tags:
            random.shuffle(tags)
        output = text_template.replace("[filewords]", ','.join(tags))
        return output


class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, concepts: [Concept], model_dir: str, shuffle_tags: bool):
        c_idx = 0
        prompts = []
        self.with_prior = False
        for concept in concepts:
            cur_class_images = 0
            print(f"Checking concept: {concept}")
            text_getter = FilenameTextGetter(shuffle_tags)
            print(f"Concept requires {concept.num_class_images} images.")

            if concept.num_class_images > 0:
                self.with_prior = True
                class_images_dir = concept["class_data_dir"]
                if class_images_dir == "" or class_images_dir is None or class_images_dir == shared.script_path:
                    class_images_dir = os.path.join(model_dir, f"classifiers_{c_idx}")
                    print(f"Class image dir is not set, defaulting to {class_images_dir}")
                class_images_dir = Path(class_images_dir)
                class_images_dir.mkdir(parents=True, exist_ok=True)
                from extensions.sd_dreambooth_extension.dreambooth.utils import get_images
                class_images = get_images(class_images_dir)
                for _ in class_images:
                    cur_class_images += 1
                print(f"Class dir {class_images_dir} has {cur_class_images} images.")
                if cur_class_images < concept.num_class_images:
                    num_new_images = concept.num_class_images - cur_class_images
                    instance_images = get_images(concept.instance_data_dir)
                    filename_texts = [text_getter.read_text(x) for x in instance_images]

                    for i in range(num_new_images):
                        text = filename_texts[i % len(filename_texts)]
                        prompt = text_getter.create_text(concept.class_prompt, text, concept.instance_token,
                                                         concept.class_token)
                        pd = PromptData()
                        pd.prompt = prompt
                        pd.negative_prompt = concept.class_negative_prompt
                        pd.steps = concept.class_infer_steps
                        pd.scale = concept.class_guidance_scale
                        pd.out_dir = class_images_dir
                        prompts.append(pd)
        random.shuffle(prompts)
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index) -> PromptData:
        return self.prompts[index]


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

        devices.torch_gc()

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
                        logging_dir=os.path.join(config.model_dir, "logging"),
                        cpu=config.use_cpu
                    )
                    self.accelerator = accelerator
                    self.del_accelerator = True
                except Exception as e:
                    if "AcceleratorState" in str(e):
                        msg = "Change in precision detected, please restart the webUI entirely to use new precision."
                    else:
                        msg = f"Exception initializing accelerator: {e}"
                    print(msg)
            print("Setting up diffusers image generator...")
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
                try:
                    no_safe = shared.cmd_opts.disable_safe_unpickle
                except:
                    no_safe = False
                shared.cmd_opts.disable_safe_unpickle = True
                accelerator.load_state(new_hotness)
                shared.cmd_opts.disable_safe_unpickle = no_safe
            if config.use_lora and lora_model is not None and lora_model != "":
                apply_lora_weights(lora_model, self.image_pipe.unet, self.image_pipe.text_encoder, lora_weight,
                                   lora_txt_weight,
                                   accelerator.device)
            print("Diffusers model configured.")
        else:
            print("Loading SD model.")
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
            print("SD model loaded.")

    def generate_images(self, prompt_data: list[PromptData]) -> [Image]:
        def update_latent(step: int, timestep: int, latents: torch.FloatTensor):
            dream_state.status.sampling_step = step
            decoded = self.image_pipe.decode_latents(latents)
            dream_state.status.current_latent = self.image_pipe.numpy_to_pil(decoded)

        positive_prompts = []
        negative_prompts = []
        seed = -1
        scale = 7.5
        steps = 60
        for prompt in prompt_data:
            positive_prompts.append(prompt.prompt)
            negative_prompts.append(prompt.negative_prompt)
            scale = prompt.scale
            steps = prompt.steps
            seed = prompt.seed
        if self.use_txt2img:
            p = StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                prompt=positive_prompts,
                negative_prompt=negative_prompts,
                batch_size=self.batch_size,
                steps=steps,
                cfg_scale=scale,
                width=self.resolution,
                height=self.resolution,
                do_not_save_grid=True,
                do_not_save_samples=True,
                do_not_reload_embeddings=True
            )
            processed = process_txt2img(p)
            print(f"Processed: {positive_prompts}")
            p.close()
            output = processed
        else:
            preview_every = shared.opts.show_progress_every_n_steps

            with self.accelerator.autocast(), torch.inference_mode():
                if seed is None or seed == '' or seed == -1:
                    seed = int(random.randrange(21474836147))
                g_cuda = torch.Generator(device=self.accelerator.device).manual_seed(seed)

                if preview_every > 0:
                    output = self.image_pipe(
                        positive_prompts,
                        num_inference_steps=steps,
                        guidance_scale=scale,
                        height=self.resolution,
                        width=self.resolution,
                        callback_steps=preview_every,
                        callback=update_latent,
                        generator=g_cuda,
                        negative_prompt=negative_prompts).images

                else:
                    output = self.image_pipe(
                        positive_prompts,
                        num_inference_steps=steps,
                        guidance_scale=scale,
                        height=self.resolution,
                        width=self.resolution,
                        generator=g_cuda,
                        negative_prompt=negative_prompts).images
        return output

    def unload(self):
        # If we have an image pipe, delete it
        if self.image_pipe is not None:
            del self.image_pipe
        if self.del_accelerator:
            del self.accelerator
        # If there was a model loaded already, reload it
        if self.last_model is not None:
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

            if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
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
    print("Generating prompts.")
    dream_state.status.job_count = 4
    from extensions.sd_dreambooth_extension.dreambooth.SuperDataset import SuperDataset
    if model_dir is None or model_dir == "":
        return "Please select a model."
    config = from_file(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(config.pretrained_model_name_or_path, "tokenizer"),
        revision=config.revision,
        use_fast=False,
    )
    dream_state.status.job_no = 1
    dream_state.status.textinfo = "Building dataset from existing files..."
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
    dream_state.status.job_no = 2
    dream_state.status.textinfo = "Appending instance and class prompts from existing files..."
    for i in range(train_dataset.__len__()):
        item = train_dataset.__getitem__(i)
        output["instance_prompts"].append(item["instance_prompt"])
        if "class_prompt" in item:
            output["existing_class_prompts"].append(item["class_prompt"])
    sample_prompts = train_dataset.get_sample_prompts()
    for prompt in sample_prompts:
        output["sample_prompts"].append(prompt.prompt)

    dream_state.status.job_no = 3
    dream_state.status.textinfo = "Building dataset for 'new' class images..."
    for concept in config.concepts_list:
        c_idx = 0
        class_images_dir = Path(concept["class_data_dir"])
        if class_images_dir == "" or class_images_dir is None or class_images_dir == shared.script_path:
            class_images_dir = os.path.join(config.model_dir, f"classifiers_{c_idx}")
            print(f"Class image dir is not set, defaulting to {class_images_dir}")
        class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = len(get_images(class_images_dir))
        if cur_class_images < concept.num_class_images:
            sample_dataset = PromptDataset(config.concepts_list, config.model_dir, config.shuffle_tags)
            for i in range(sample_dataset.__len__()):
                prompt = sample_dataset.__getitem__(i)
                output["new_class_prompts"].append(prompt.prompt)
        c_idx += 1
    dream_state.status.job_no = 4
    dream_state.status.textinfo = "Prompt generation complete."
    return json.dumps(output)


def generate_classifiers(args: DreamboothConfig, lora_model: str = "", lora_weight: float = 1.0,
                         lora_text_weight: float = 1.0, use_txt2img: bool = True, accelerator: Accelerator = None):
    """

    @param args: A DreamboothConfig
    @param lora_model: Optional path to a lora model to use. You probably don't want to use this.
    @param lora_weight: Alpha to use when merging lora unet.
    @param lora_text_weight: Alpha to use when merging lora text encoder.
    @param use_txt2img: Generate images using txt2image. Does not use lora.
    @param accelerator: An optional existing accelerator to use.
    @return:
    generated: Number of images generated
    with_prior: Whether prior preservation should be used
    images: A list of strings with paths to images.
    """
    printm("Generating class images...")
    out_images = []
    try:
        prompt_dataset = PromptDataset(args.concepts_list, args.model_dir, args.shuffle_tags)
    except Exception as p:
        print(f"Exception generating dataset: {str(p)}")
        return 0, False

    with_prior = prompt_dataset.with_prior

    set_len = prompt_dataset.__len__()
    if set_len == 0:
        print("Nothing to generate.")
        return 0, prompt_dataset.with_prior, []

    dream_state.status.textinfo = f"Generating {set_len} class images for training..."
    dream_state.status.job_count = set_len
    dream_state.status.job_no = 0
    print(f"Creating image builder {args.sample_batch_size}...")
    builder = ImageBuilder(args, use_txt2img=use_txt2img, lora_model=lora_model, lora_weight=lora_weight,
                           lora_txt_weight=lora_text_weight, batch_size=args.sample_batch_size, accelerator=accelerator)
    generated = 0
    pbar = tqdm(total=set_len - 1)

    for i in range(set_len):
        if dream_state.status.interrupted:
            break
        prompts = []
        for b in range(args.sample_batch_size):
            pd = prompt_dataset.__getitem__(i)
            prompts.append(pd)

        new_images = builder.generate_images(prompts)
        i_idx = 0
        for image in new_images:
            try:
                pd = prompts[i_idx]
                image_base = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = os.path.join(pd.out_dir, f"{image_base}.png")
                print(f"Trying to save: {image_filename}")
                image.save(image_filename)
                out_images.append(image_filename)
                txt_filename = image_filename.replace(".png", ".txt")
                with open(txt_filename, "w", encoding="utf8") as file:
                    file.write(pd.prompt)
                dream_state.status.job_no += 1
                dream_state.status.textinfo = f"Class image {i}/{set_len}, " \
                                              f"Prompt: '{pd.prompt}'"
            except Exception as e:
                print(f"Exception generating images: {e}")
                traceback.print_exc()
            i_idx += 1
            dream_state.status.current_image = image
            if pbar is not None:
                pbar.update()
            generated += 1

        dream_state.status.current_image = images.image_grid(new_images)
    builder.unload()
    del prompt_dataset
    cleanup()
    printm(f"Generated {generated} new class images.")
    return generated, with_prior, out_images


# Implementation from https://github.com/bmaltais/kohya_ss
def encode_hidden_state(text_encoder: CLIPTextModel, input_ids, pad_tokens, b_size, max_token_length,
                        tokenizer_max_length):
    if pad_tokens:
        input_ids = input_ids.reshape((-1, tokenizer_max_length))  # batch_size*3, 77

    clip_skip = shared.opts.CLIP_stop_at_last_layers
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
