from pydantic import Field
from finetune.dataclasses.base_config import BaseConfig


class TrainDreamboothLoraSdxlConfig(BaseConfig):
    adam_beta1: float = Field(default=0.9, title='Adam Beta1', description='The beta1 parameter for the Adam and Prodigy optimizers.', group='Optimizer', advanced=True)
    adam_beta2: float = Field(default=0.999, title='Adam Beta2', description='The beta2 parameter for the Adam and Prodigy optimizers.', group='Optimizer', advanced=True)
    adam_epsilon: float = Field(default=1e-08, title='Adam Epsilon', description='Epsilon value for the Adam optimizer and Prodigy optimizers.', group='Optimizer', advanced=True)
    adam_weight_decay: float = Field(default=0.01, title='Adam Weight Decay', description='Weight decay to use for unet params', group='Optimizer', advanced=True)
    adam_weight_decay_text_encoder: float = Field(default=0.001, title='Adam Weight Decay Text Encoder', description='Weight decay to use for text_encoder', group='Optimizer', advanced=True)
    allow_tf32: bool = Field(default=True, title='Allow Tf32', description='Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices', group='Performance', advanced=True)
    cache_dir: str = Field(default=None, title='Cache Dir', description='The directory where the downloaded models and datasets will be stored.', ignore=True)
    caption_column: str = Field(default='text', title='Caption Column', description='The column of the dataset containing the instance prompt for each image', group='Dataset')
    center_crop: bool = Field(default=False, title='Center Crop', description='Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.', group='Image Processing', advanced=True)
    checkpointing_steps: int = Field(default=500, title='Checkpointing Steps', description='Save a checkpoint of the training state every X updates. These checkpoints can be used both as final checkpoints in case they are better than the last checkpoint, and are also suitable for resuming training using `--resume_from_checkpoint`.', group='Intervals', min=0, max=100000)
    checkpoints_total_limit: str = Field(default=3, title='Checkpoints Total Limit', description='Max number of checkpoints to store.', group='Saving', min=0, max=100, advanced=True)
    class_data_dir: str = Field(default=None, title='Class Data Dir', description='A folder containing the training data of class images.', group='Dataset')
    class_prompt: str = Field(default=None, title='Class Prompt', description='The prompt to specify images in the same class as provided instance images.', group='Dataset')
    crops_coords_top_left_h: int = Field(default=0, title='Crops Coords Top Left H', description='Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet.', group='Dataset', advanced=True, min=0, max=100000, step=1)
    crops_coords_top_left_w: int = Field(default=0, title='Crops Coords Top Left W', description='Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet.', group='Dataset', advanced=True, min=0, max=100000, step=1)
    dataloader_num_workers: int = Field(default=1, title='Dataloader Num Workers', description='Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.', group='Performance', min=0, max=100, advanced=True)
    dataset_config_name: str = Field(default=None, title='Dataset Config Name', description="The config of the Dataset, leave as None if there's only one config.", group='Dataset', advanced=True)
    dataset_name: str = Field(default=None, title='Dataset Name', description='The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private, dataset). It can also be a path pointing to a local copy of a dataset in your filesystem, or to a folder containing files that 🤗 Datasets can understand.', group='Dataset')
    enable_xformers_memory_efficient_attention: bool = Field(default=True, title='Enable Xformers Memory Efficient Attention', description='Whether or not to use xformers.', group='Performance')
    gradient_accumulation_steps: int = Field(default=1, title='Gradient Accumulation Steps', description='Number of updates steps to accumulate before performing a backward/update pass.', group='Batching', advanced=True)
    gradient_checkpointing: bool = Field(default=True, title='Gradient Checkpointing', description='Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.', group='Performance', advanced=True)
    hub_model_id: str = Field(default=None, title='Hub Model Id', description='The name of the repository to keep in sync with the local `output_dir`.', group='Saving', visible=False)
    hub_token: str = Field(default=None, title='Hub Token', description='The token to use to push to the Model Hub.', group='Saving', visible=False)
    image_column: str = Field(default='image', title='Image Column', description="The column of the dataset containing the target image. By default, the standard Image Dataset maps out 'file_name' to 'image'.", group='Dataset')
    instance_data_dir: str = Field(default=None, title='Instance Data Dir', description='A folder containing the training data. ', group='Dataset')
    instance_prompt: str = Field(default=None, title='Instance Prompt', description="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'", group='Dataset')
    learning_rate: float = Field(default=5e-06, title='Learning Rate', description='Initial learning rate (after the potential warmup period) to use.', group='Learning Rate')
    local_rank: int = Field(default=-1, title='Local Rank', description='For distributed training: local_rank', group='Performance', advanced=True)
    logging_dir: str = Field(default='logs', title='Logging Dir', description='[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.', group='Logging', visible=False)
    lr_num_cycles: int = Field(default=1, title='Lr Num Cycles', description='Number of hard resets of the lr in cosine_with_restarts scheduler.', group='Learning Rate', min=0, max=100000, advanced=True)
    lr_power: float = Field(default=1.0, title='Lr Power', description='Power factor of the polynomial scheduler.', group='Learning Rate', advanced=True, min=0, max=1, step=0.1)
    lr_scheduler: str = Field(default='constant', title='Lr Scheduler', description='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]', group='Learning Rate', choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'], advanced=True)
    lr_warmup_steps: int = Field(default=500, title='Lr Warmup Steps', description='Number of steps for the warmup in the lr scheduler.', group='Learning Rate', min=0, max=100000, advanced=True)
    max_grad_norm: float = Field(default=1.0, title='Max Grad Norm', description='Max gradient norm.', group='Optimizer', advanced=True, min=0, max=1, step=0.1)
    max_train_steps: str = Field(default=None, title='Max Train Steps', description='Total number of training steps to perform.  If provided, overrides num_train_epochs.', group='Intervals', visible=False)
    mixed_precision: str = Field(default='bf16', title='Mixed Precision', description='Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.', choices=['no', 'fp16', 'bf16'], group='Performance', advanced=True)
    num_class_images: int = Field(default=100, title='Num Class Images', description='Minimal class images for prior preservation loss. If there are not enough images already present in class_data_dir, additional images will be sampled with class_prompt.', group='Dataset', min=0, max=100000)
    num_train_epochs: int = Field(default=1, title='Num Train Epochs', description='', group='Intervals', min=0, max=100000)
    num_validation_images: int = Field(default=4, title='Num Validation Images', description='Number of images that should be generated during validation with `validation_prompt`.', group='Validation', advanced=True, min=0, max=100)
    optimizer: str = Field(default='AdamW', title='Optimizer', description='The optimizer type to use. Choose between ["AdamW", "prodigy"]', choices=['AdamW', 'prodigy'], group='Optimizer', advanced=True)
    output_dir: str = Field(default='t2iadapter-model', title='Output Dir', description='The output directory where the model predictions and checkpoints will be written.', ignore=True)
    pretrained_model_name_or_path: str = Field(default=None, title='Pretrained Model Name Or Path', description='Path to pretrained model or model identifier from huggingface.co/models.', ignore=True)
    pretrained_vae_model_name_or_path: str = Field(default=None, title='Pretrained Vae Model Name Or Path', description='Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.', group='Model', advanced=True)
    prior_generation_precision: str = Field(default='bf16', title='Prior Generation Precision', description='Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32.', choices=['no', 'fp32', 'fp16', 'bf16'], group='Performance', advanced=True)
    prior_loss_weight: float = Field(default=1.0, title='Prior Loss Weight', description='The weight of prior preservation loss.', group='Dataset', advanced=True, min=0, max=1, step=0.1)
    prodigy_beta3: str = Field(default=None, title='Prodigy Beta3', description='coefficients for computing the Prodidy stepsize using running averages. If set to None, uses the value of square root of beta2. Ignored if optimizer is adamW', group='Optimizer', advanced=True)
    prodigy_decouple: bool = Field(default=True, title='Prodigy Decouple', description='Use AdamW style decoupled weight decay', group='Optimizer', advanced=True)
    prodigy_safeguard_warmup: bool = Field(default=True, title='Prodigy Safeguard Warmup', description='Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. Ignored if optimizer is adamW', group='Optimizer', advanced=True)
    prodigy_use_bias_correction: bool = Field(default=True, title='Prodigy Use Bias Correction', description="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW", group='Optimizer', advanced=True)
    push_to_hub: bool = Field(default=False, title='Push To Hub', description='Whether or not to push the model to the Hub.', group='Saving', visible=False)
    rank: int = Field(default=4, title='Rank', description='The dimension of the LoRA update matrices.', group='LORA', advanced=True)
    repeats: int = Field(default=1, title='Repeats', description='How many times to repeat the training data.', group='Dataset', advanced=True)
    report_to: str = Field(default='tensorboard', title='Report To', description='The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.', choices=['all', 'tensorboard', 'wandb'], group='Logging')
    resolution: int = Field(default=1024, title='Resolution', description='The resolution for input images, all the images in the train/validation dataset will be resized to this resolution', group='Image Processing', min=256, max=4096, step=64)
    resume_from_checkpoint: str = Field(default=None, title='Resume From Checkpoint', description='Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.', ignore=True)
    revision: str = Field(default=None, title='Revision', description='Revision of pretrained model identifier from huggingface.co/models.', ignore=True)
    sample_batch_size: int = Field(default=4, title='Sample Batch Size', description='Batch size (per device) for sampling images.', group='Batching', advanced=True, min=1, max=1000)
    scale_lr: bool = Field(default=False, title='Scale Lr', description='Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.', group='Learning Rate', advanced=True)
    seed: str = Field(default=None, title='Seed', description='A seed for reproducible training.', group='Performance', advanced=True)
    snr_gamma: str = Field(default=None, title='Snr Gamma', description='SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556.', group='Performance', advanced=True)
    text_encoder_lr: float = Field(default=5e-06, title='Text Encoder Lr', description='Text encoder learning rate to use.', group='Learning Rate', advanced=True, min=0, max=1, step=1e-05)
    train_batch_size: int = Field(default=4, title='Train Batch Size', description='Batch size (per device) for the training dataloader.', group='Batching', min=1, max=1000)
    train_text_encoder: bool = Field(default=False, title='Train Text Encoder', description='Whether to train the text encoder. If set, the text encoder should be float32 precision.', group='Performance', advanced=True)
    use_8bit_adam: bool = Field(default=True, title='Use 8Bit Adam', description='Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW', group='Performance', advanced=True)
    validation_epochs: int = Field(default=1, title='Validation Epochs', description='Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images`.', group='Intervals')
    validation_prompt: str = Field(default=None, title='Validation Prompt', description='A prompt that is used during validation to verify that the model is learning.', group='Validation', advanced=True)
    variant: str = Field(default=None, title='Variant', description="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16", ignore=True)
    with_prior_preservation: bool = Field(default=False, title='With Prior Preservation', description='Flag to add prior preservation loss.', group='Dataset', advanced=True)