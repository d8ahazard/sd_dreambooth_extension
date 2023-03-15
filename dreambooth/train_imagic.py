import argparse
import os
from typing import List

import torch
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from torch.nn import functional
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dreambooth import shared
from dreambooth.dataclasses.db_config import DreamboothConfig
from dreambooth.shared import status
from dreambooth.utils.image_utils import list_features, is_image

logger = get_logger(__name__)


class TrainResult:
    config: DreamboothConfig = None
    mem_record: List = []
    msg: str = ""
    samples: [Image] = []


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        required=True,
        help="Path to input image to edit.",
    )
    parser.add_argument(
        "--target_text",
        type=str,
        default=None,
        help="The target text describing the output image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--emb_train_steps",
        type=int,
        default=500,
        help="Total number of training steps to perform.",
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
        "--emb_learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizing the embeddings.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate for fine tuning the model.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def train_imagic(args: DreamboothConfig):
    logging_dir = os.path.join(args.model_dir, "logging")
    print("Initializing imagic.")
    result = TrainResult()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer",
                                              use_auth_token=False)

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder",
                                                 use_auth_token=False)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", use_auth_token=True)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                                use_auth_token=False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    optimizer_class = torch.optim.Adam
    if args.optimizer == "8bit AdamW":
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.Adam8bit
        except:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    pil_features = list_features()
    concepts = args.concepts()
    concept = concepts[0]
    instance_dir = concept.instance_data_dir
    input_image = None

    instance_dir = os.path.abspath(instance_dir)
    for file in os.listdir(instance_dir):
        check = os.path.join(instance_dir, file)
        if is_image(check, pil_features):
            input_image = Image.open(check).convert("RGB")
            break

    # Encode the input images.
    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    init_latents = None
    if input_image is not None:
        init_image = image_transforms(input_image)
        init_image = init_image[None].to(device=accelerator.device, dtype=weight_dtype)

        with torch.inference_mode():
            init_latents = vae.encode(init_image).latent_dist.sample()
            init_latents = 0.18215 * init_latents

    # Encode the target text.
    text_ids = tokenizer(
        concept.instance_token,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids

    text_ids = text_ids.to(device=accelerator.device)
    with torch.inference_mode():
        target_embeddings = text_encoder(text_ids)[0]

    del vae, text_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_embeddings = target_embeddings.float()
    optimized_embeddings = target_embeddings.clone()

    # Optimize the text embeddings first.
    optimized_embeddings.requires_grad_(True)
    optimizer = optimizer_class(
        [optimized_embeddings],  # only optimize embeddings
        lr=1e-3
    )

    unet, optimizer = accelerator.prepare(unet, optimizer)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers will initialize automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("imagic")

    def train_loop(pbar, optim):
        for step in pbar:
            loss_total = 0
            if status.interrupted:
                break
            status.job_no += 1
            with accelerator.accumulate(unet):
                noise = torch.randn_like(init_latents)
                # Not sure if this matters, but technically, add_noise should take a FloatTensor
                noise = noise.float()
                bsz = init_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, 1000, (bsz,),
                                          device=init_latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(init_latents, noise, timesteps)

                noise_pred = unet(noisy_latents, timesteps, optimized_embeddings).sample

                current_loss = functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                loss_total += current_loss
                avg_loss = loss_total / (step + 1)
                accelerator.backward(current_loss)
                optim.step()
                optim.zero_grad(set_to_none=True)

            if not step % 10:
                logs = {"loss": avg_loss}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=step)

        accelerator.wait_for_everyone()

    status.job_count = args.num_train_epochs * 2
    progress_bar = tqdm(range(args.num_train_epochs), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Optimizing embedding")
    status.textinfo = "Optimizing Embedding"
    print(status.textinfo)
    train_loop(progress_bar, optimizer)

    optimized_embeddings.requires_grad_(False)
    if accelerator.is_main_process:
        status.textinfo = "Saving embedding(s)."
        print(status.textinfo)
        emb_dir = shared.embeddings_dir
        torch.save(target_embeddings.cpu(), os.path.join(emb_dir, f"{args.model_name}.pt"))
        torch.save(optimized_embeddings.cpu(), os.path.join(emb_dir, f"{args.model_name}_optimized.pt"))

    # Fine tune the diffusion model.
    optimizer = optimizer_class(
        accelerator.unwrap_model(unet).parameters(),
        lr=args.learning_rate
    )

    optimizer = accelerator.prepare(optimizer)
    progress_bar = tqdm(range(args.num_train_epochs), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Fine Tuning")
    status.textinfo = "Fine Tuning"
    print(status.textinfo)
    unet.train()

    train_loop(progress_bar, optimizer)

    # Create the pipeline using the trained modules and save it.
    if accelerator.is_main_process:
        status.textinfo = "Saving pretrained model."
        print(status.textinfo)
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            use_auth_token=True
        )
        pipeline.save_pretrained(args.pretrained_model_name_or_path, safe_serialization=True)

    accelerator.end_training()
    print("Training complete")
    result.msg = "IMagic training complete."
    result.samples = []
    return result
