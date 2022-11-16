# From shivam shiaro's repo, with "minimal" modification to hopefully allow for smoother updating?
import argparse
import gc
import hashlib
import itertools
import logging
import math
import os
import random
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models import attention
from diffusers.optimization import get_scheduler
from dreambooth.dreambooth import dumb_safety, save_checkpoint, list_features, is_image, printm
from dreambooth.finetune_utils import FilenameTextGetter, EMAModel, encode_hidden_state
from huggingface_hub import HfFolder, whoami
from torch import autocast
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dreambooth import xattention
from modules import shared, devices
from modules.images import sanitize_filename_part

# Custom stuff
try:
    cmd_dreambooth_models_path = shared.cmd_opts.dreambooth_models_path
except:
    cmd_dreambooth_models_path = None

pil_features = list_features()
mem_record = {}
attention.CrossAttention = xattention.CrossAttention
attention.Transformer2DModel = xattention.Transformer2DModelOutput
# End custom stuff

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.setLevel(logging.DEBUG)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--save_sample_prompt",
        type=str,
        default=None,
        help="The prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--save_sample_negative_prompt",
        type=str,
        default=None,
        help="The negative prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--n_save_sample",
        type=int,
        default=4,
        help="The number of samples to save.",
    )
    parser.add_argument(
        "--save_guidance_scale",
        type=float,
        default=7.5,
        help="CFG for save sample.",
    )
    parser.add_argument(
        "--save_infer_steps",
        type=int,
        default=50,
        help="The number of inference steps for save sample.",
    )
    parser.add_argument(
        "--pad_tokens",
        default=False,
        action="store_true",
        help="Flag to pad tokens to length 77.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--shuffle_after_epoch",
        default=False,
        action="store_true",
        help="Whether or not to shuffle and recache training dataset after every epoch."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=-1, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--save_interval", type=int, default=10_000, help="Save weights every N steps.")
    parser.add_argument("--save_min_steps", type=int, default=0, help="Start saving weights after N steps.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--not_cache_latents", action="store_true", help="Do not precompute and cache latents from VAE.")
    parser.add_argument("--hflip", action="store_true", help="Apply horizontal flip data augmentation.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            concepts_list,
            tokenizer,
            with_prior_preservation=True,
            size=512,
            center_crop=False,
            num_class_images=None,
            pad_tokens=False,
            hflip=False,
            max_token_length=75,
            shuffle_after_epoch=False                                     
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.with_prior_preservation = with_prior_preservation
        self.pad_tokens = pad_tokens
        self.shuffle_after_epoch = shuffle_after_epoch
        self.instance_images_path = []
        self.class_images_path = []
        self.class_images_randomizer_stack = []

        self.max_token_length = max_token_length
        self.tokenizer_max_length = self.tokenizer.model_max_length if max_token_length == 75 else max_token_length + 2
        self.text_getter = FilenameTextGetter()
        for concept in concepts_list:
            inst_img_path = [(x, concept["instance_prompt"], self.text_getter.read_text(x)) for x in
                             Path(concept["instance_data_dir"]).iterdir() if is_image(x, pil_features)]
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_img_path = [(x, concept["class_prompt"], self.text_getter.read_text(x)) for x in
                                  Path(concept["class_data_dir"]).iterdir() if is_image(x, pil_features)]
                self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)

        self.image_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5 * hflip),
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def tokenize(self, text):
        if not self.pad_tokens:
            input_ids = self.tokenizer(text, padding="do_not_pad", truncation=True,
                                       max_length=self.tokenizer.model_max_length).input_ids
            return input_ids

        input_ids = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.tokenizer_max_length,
                                   return_tensors="pt").input_ids
        if self.tokenizer_max_length > self.tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            for i in range(1, self.tokenizer_max_length - self.tokenizer.model_max_length + 2,
                           self.tokenizer.model_max_length - 2):
                iid = (input_ids[0].unsqueeze(0),
                       input_ids[i:i + self.tokenizer.model_max_length - 2],
                       input_ids[-1].unsqueeze(0))
                iid = torch.cat(iid)
                iids_list.append(iid)
            input_ids = torch.stack(iids_list)  # 3,77

        return input_ids

    def __len__(self):
        return self._length
    def _get_random_class_image_index(self):
        if len(self.class_images_randomizer_stack) == 0:
            self.class_images_randomizer_stack = [x for x in range(self.num_class_images)]
        random_index = random.randint(0, len(self.class_images_randomizer_stack) - 1)
        result = self.class_images_randomizer_stack.pop(random_index)
        return result
    def __getitem__(self, index):
        example = {}
        instance_path, instance_prompt, instance_text = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt"] = self.text_getter.create_text(instance_prompt,
                                                                  instance_text)  # TODO: show the final prompt of the image currently being trained in the ui
        example["instance_prompt_ids"] = self.tokenize(example["instance_prompt"])

        if self.with_prior_preservation:

            if not self.shuffle_after_epoch:
                class_path, class_prompt, class_text = self.class_images_path[index % self.num_class_images]
            else:
                class_path, class_prompt, class_text = self.class_images_path[self._get_random_class_image_index()]
                
            class_image = Image.open(class_path)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.text_getter.create_text(class_prompt, class_text)
            example["class_prompt_ids"] = self.tokenize(example["class_prompt"])

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples, filename_texts):
        self.prompt = prompt
        self.num_samples = num_samples
        self.filename_texts = filename_texts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["filename_text"] = self.filename_texts[index % len(self.filename_texts)] if len(
            self.filename_texts) > 0 else ""
        example["prompt"] = self.prompt.replace("[filewords]", example["filename_text"])
        example["index"] = index
        return example


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index]


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args, memory_record):
    args.tokenizer_name = None
    global mem_record
    mem_record = memory_record
    logging_dir = Path(args.output_dir, "logging")
    args.max_token_length = int(args.max_token_length)
    if not args.pad_tokens and args.max_token_length > 75:
        print("Cannot raise token length limit above 75 when pad_tokens=False")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
        cpu=args.use_cpu
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        msg = "Gradient accumulation is not supported when training the text encoder in distributed training. " \
              "Please set gradient_accumulation_steps to 1. This feature will be supported in the future. Text " \
              "encoder training will be disabled."
        logger.debug(msg)
        shared.state.textinfo = msg
        args.train_text_encoder = False

    if args.seed is not None and args.seed != -1 and args.seed != "":
        set_seed(args.seed)

    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "sample_prompt": args.save_sample_prompt,
                "negative_prompt": args.save_sample_negative_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir
            }
        ]

    if args.with_prior_preservation:
        pipeline = None
        text_getter = FilenameTextGetter()
        for concept in args.concepts_list:
            class_images_dir = Path(concept["class_data_dir"])
            class_images_dir.mkdir(parents=True, exist_ok=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < args.num_class_images:
                shared.state.textinfo = f"Generating class images for training..."
                torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                if pipeline is None:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=AutoencoderKL.from_pretrained(
                            args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
                            subfolder=None if args.pretrained_vae_name_or_path else "vae",
                            revision=args.revision,
                            torch_dtype=torch_dtype
                        ),
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        revision=args.revision
                    )
                    pipeline.safety_checker = dumb_safety
                    pipeline.set_progress_bar_config(disable=True)
                    pipeline.to(accelerator.device)

                num_new_images = args.num_class_images - cur_class_images
                logger.debug(f"Number of class images to sample: {num_new_images}.")
                shared.state.job_count = num_new_images
                shared.state.job_no = 0
                save_txt = "[filewords]" in concept["class_prompt"]
                filename_texts = [text_getter.read_text(x) for x in Path(concept["instance_data_dir"]).iterdir() if
                                  is_image(x, pil_features)]
                sample_dataset = PromptDataset(concept["class_prompt"], num_new_images, filename_texts)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

                sample_dataloader = accelerator.prepare(sample_dataloader)

                with torch.autocast("cuda"), torch.inference_mode():
                    for example in tqdm(
                            sample_dataloader, desc="Generating class images",
                            disable=not accelerator.is_local_main_process
                    ):
                        print(f"Example prompt is: {example['prompt']}, negative is {args.class_negative_prompt}")
                        images = pipeline(example["prompt"][0], num_inference_steps=args.class_infer_steps,
                                          guidance_scale=args.class_guidance_scale,
                                          negative_prompt=args.class_negative_prompt).images

                        for i, image in enumerate(images):
                            if shared.state.interrupted:
                                print("Generation canceled.")
                                return args, "Training canceled."
                            shared.state.job_no += 1
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                            shared.state.current_image = image
                            image.save(image_filename)
                            if save_txt:
                                txt_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.txt"
                                with open(txt_filename, "w", encoding="utf8") as file:
                                    # we have to write filename_text and not full prompt here, otherwise "dog, [filewords]" becomes "dog, dog, [filewords]" when read. Any elegant solution?
                                    file.write(example["filename_text"][i] + "\n")

        del pipeline
        del text_getter
        devices.torch_gc()

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        torch_dtype=torch.float32
    )
    printm("Loaded model.")

    def create_vae(device, weight_dtype):
        result = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
        )
        result.requires_grad_(False)
        result.to(device, dtype=weight_dtype)
        return result

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = create_vae(accelerator.device, weight_dtype)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    use_adam = False
    optimizer_class = torch.optim.AdamW

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            use_adam = True
        except Exception as a:
            logger.warn(f"Exception importing 8bit adam: {a}")

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    def cleanup_memory():
        try:
            printm("CLEANUP: ")
            if unet:
                del unet
            if text_encoder:
                del text_encoder
            if tokenizer:
                del tokenizer
            if optimizer:
                del optimizer
            if train_dataloader:
                del train_dataloader
            if train_dataset:
                del train_dataset
            if lr_scheduler:
                del lr_scheduler
            if vae:
                del vae
            if ema_unet:
                del ema_unet
        except:
            pass
        try:
            gc.collect()  # Python thing
            torch.cuda.empty_cache()  # PyTorch thing
        except:
            pass
        printm("Cleanup Complete.")

    train_dataset = DreamBoothDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        pad_tokens=args.pad_tokens,
        hflip=args.hflip,
        max_token_length=args.max_token_length,
        shuffle_after_epoch=args.shuffle_after_epoch                                                    
    )

    if train_dataset.num_instance_images == 0:
        msg = "Please provide a directory with actual images in it."
        logger.debug('msg')
        cleanup_memory()
        return args, ""

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        if not args.pad_tokens:
            input_ids = tokenizer.pad(
                {"input_ids": input_ids},
                padding=True,
                return_tensors="pt",
            ).input_ids
        else:
            input_ids = torch.stack(input_ids)

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True
    )

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.

    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    def cache_latents(train_dataset=None, train_dataloader=None, vae=None):
        if train_dataset is not None:
            del train_dataset
        if train_dataloader is not None:
            del train_dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        train_dataset = DreamBoothDataset(
            concepts_list=args.concepts_list,
            tokenizer=tokenizer,
            with_prior_preservation=args.with_prior_preservation,
            size=args.resolution,
            center_crop=args.center_crop,
            num_class_images=args.num_class_images,
            pad_tokens=args.pad_tokens,
            hflip=args.hflip,
            max_token_length=args.max_token_length,
            shuffle_after_epoch=args.shuffle_after_epoch
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True
        )
        latents_cache = []
        text_encoder_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True,
                                                                 dtype=weight_dtype)
                batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
                latent_dist = vae.encode(batch["pixel_values"]).latent_dist
                b_size = latent_dist.sample().shape[
                    0]  # TODO: the sample is not necessary, but I don't know torch, could use a little help here
                latents_cache.append(latent_dist)
                if args.train_text_encoder:
                    text_encoder_cache.append(batch["input_ids"])
                else:
                    text_encoder_cache.append(
                        encode_hidden_state(text_encoder, batch["input_ids"], args.pad_tokens, b_size,
                                            args.max_token_length, tokenizer.model_max_length))
        train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x,
                                                       shuffle=True)

        del vae
        vae = None
        if not args.train_text_encoder:
            del text_encoder
            text_encoder = None
        devices.torch_gc()

        return train_dataset, train_dataloader

    if not args.not_cache_latents:
        train_dataset, train_dataloader = cache_latents(vae=vae)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    printm("Scheduler Loaded")

    # create ema, fix OOM
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())
        ema_unet.to(accelerator.device, dtype=weight_dtype)
        # ema_unet.to(accelerator.device)
        if args.train_text_encoder and text_encoder is not None:
            unet, ema_unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, ema_unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, ema_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, ema_unet, optimizer, train_dataloader, lr_scheduler
            )
    else:
        ema_unet = None
        if args.train_text_encoder and text_encoder is not None:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    stats = f"CPU: {args.use_cpu} Adam: {use_adam}, Prec: {args.mixed_precision}, " \
            f"Prior: {args.with_prior_preservation}, Grad: {args.gradient_checkpointing}, " \
            f"TextTr: {args.train_text_encoder} "

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Shuffle After Epoch = {args.shuffle_after_epoch}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    def save_weights(step, save_model, save_img):
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            if args.train_text_encoder:
                text_enc_model = accelerator.unwrap_model(text_encoder)
            else:
                text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
            if args.use_ema:
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=text_enc_model,
                vae=AutoencoderKL.from_pretrained(
                    args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
                    subfolder=None if args.pretrained_vae_name_or_path else "vae"
                ),
                scheduler=scheduler,
                torch_dtype=torch.float16,
                revision=args.revision,
                safety_checker=None
            )
            pipeline = pipeline.to("cuda")
            with autocast("cuda"), torch.inference_mode():
                if save_model:
                    shared.state.textinfo = f"Saving checkpoint at step {args.revision}..."
                    try:
                        pipeline.save_pretrained(args.pretrained_model_name_or_path)
                        save_checkpoint(args.model_name, args.pretrained_vae_name_or_path, args.revision,
                                        args.mixed_precision == "fp16")
                        if args.use_ema:
                            ema_unet.restore(unet.parameters())
                    except Exception as e:
                        logger.debug(f"Exception saving checkpoint/model: {e}")
                        traceback.print_exc()
                        pass
                save_dir = args.output_dir
                if args.save_sample_prompt is not None and save_img:
                    shared.state.textinfo = f"Saving preview image at step {args.revision}..."
                    try:
                        pipeline = pipeline.to(accelerator.device)
                        seed = args.seed
                        # I feel like this might not actually be necessary...but what the heck.
                        if seed is None or seed == '' or seed == -1:
                            seed = int(random.randrange(4294967294))
                        g_cuda = torch.Generator(device=accelerator.device).manual_seed(seed)
                        pipeline.set_progress_bar_config(disable=True)
                        sample_dir = os.path.join(save_dir, "samples")
                        os.makedirs(sample_dir, exist_ok=True)
                        with torch.autocast("cuda"), torch.inference_mode():
                            for c in args.concepts_list:
                                sample_prompt = args.save_sample_prompt
                                negative_prompt = args.save_sample_negative_prompt
                                if "sample_prompt" in c:
                                    sample_prompt = c["sample_prompt"]
                                if "negative_prompt" in c:
                                    negative_prompt = c["sample_prompt"]
                                if sample_prompt is None or sample_prompt == "":
                                    sample_prompt = c["instance_prompt"]
                                for i in tqdm(range(args.n_save_sample), desc="Generating samples"):
                                    images = pipeline(
                                        sample_prompt,
                                        negative_prompt=negative_prompt,
                                        guidance_scale=args.save_guidance_scale,
                                        num_inference_steps=args.save_infer_steps,
                                        generator=g_cuda
                                    ).images
                                    shared.state.current_image = images[0]
                                    sanitized_prompt = sanitize_filename_part(args.save_sample_prompt, replace_spaces=False)
                                    images[0].save(os.path.join(sample_dir, f"{sanitized_prompt}{step}-{i}.png"))
                    except Exception as e:
                        logger.debug(f"Exception with the stupid image again: {e}")
            logger.debug(f"[*] Weights saved at {save_dir}")
            del pipeline
            del scheduler
            del text_enc_model
            devices.torch_gc()

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    lifetime_step = args.total_steps
    shared.state.job_count = args.max_train_steps
    shared.state.job_no = global_step
    shared.state.textinfo = f"Training step: {global_step}/{args.max_train_steps}"
    loss_avg = AverageMeter()
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()
    for epoch in range(args.num_train_epochs):
        try:
            unet.train()
            if args.train_text_encoder and text_encoder is not None:
                text_encoder.train()
 
            if args.shuffle_after_epoch and (global_step > len(train_dataset)):
                if vae is None:
                    vae = create_vae(accelerator.device, weight_dtype)
                train_dataset, train_dataloader = cache_latents(train_dataset, train_dataloader, vae)
                                               
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    with torch.no_grad():
                        if not args.not_cache_latents:
                            latent_dist = batch[0][0]
                        else:
                            latent_dist = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist
                        latents = latent_dist.sample() * 0.18215
                        b_size = latents.shape[0]

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                              device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    with text_enc_context:
                        if not args.not_cache_latents:
                            if args.train_text_encoder:
                                encoder_hidden_states = encode_hidden_state(text_encoder, batch[0][1], args.pad_tokens,
                                                                            b_size, args.max_token_length,
                                                                            tokenizer.model_max_length)
                            else:
                                encoder_hidden_states = batch[0][1]
                        else:
                            encoder_hidden_states = encode_hidden_state(text_encoder, batch["input_ids"],
                                                                        args.pad_tokens, b_size, args.max_token_length,
                                                                        tokenizer.model_max_length)

                    # Predict the noise residual
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if args.with_prior_preservation:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        noise, noise_prior = torch.chunk(noise, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

                        # Compute prior loss
                        prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                    accelerator.backward(loss)
                    # if accelerator.sync_gradients:
                    #     params_to_clip = (
                    #         itertools.chain(unet.parameters(), text_encoder.parameters())
                    #         if args.train_text_encoder
                    #         else unet.parameters()
                    #     )
                    #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    loss_avg.update(loss.detach_(), bsz)

                    # Update EMA
                    if args.use_ema and ema_unet is not None:
                        ema_unet.step(unet.parameters())

                if not global_step % 10:
                    logs = {"loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                progress_bar.update(1)
                global_step += 1
                args.revision += 1
                shared.state.job_no = global_step

                training_complete = global_step >= args.max_train_steps or shared.state.interrupted
                if global_step > 0:
                    save_img = args.save_preview_every and not global_step % args.save_preview_every
                    save_model = args.save_embedding_every and not global_step % args.save_embedding_every
                    if training_complete:
                        save_img = True
                        save_model = True
                    if save_img or save_model:
                        save_weights(args.revision, save_model, save_img)

                if global_step == 0 or global_step == 5:
                    printm(f"Step {global_step} completed.")
                shared.state.textinfo = f"Training, step {global_step}/{args.max_train_steps} current, {args.revision}/{args.max_train_steps + lifetime_step} lifetime"

                if training_complete:
                    logger.debug("Training complete??")
                    if shared.state.interrupted:
                        state = "cancelled"
                    else:
                        state = "complete"

                    shared.state.textinfo = f"Training {state} {global_step}/{args.max_train_steps}, {args.revision}" \
                                            f" total."

                    break
            training_complete = global_step >= args.max_train_steps or shared.state.interrupted
            accelerator.wait_for_everyone()
            if training_complete:
                break
        except Exception as m:
            printm(f"Exception while training: {m}")
            traceback.print_exc()

    cleanup_memory()
    accelerator.end_training()
    return args, mem_record