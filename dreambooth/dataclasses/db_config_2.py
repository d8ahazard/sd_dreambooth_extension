from typing import Optional, List

from pydantic import Field

from dreambooth.dataclasses.base_config import BaseConfig


def sanitize_name(name):
    return "".join(x for x in name if (x.isalnum() or x in "._- "))


class DreamboothConfig2(BaseConfig):
    config_prefix: str = Field("db2", description="Prefix for the config file.")
    model_dir: str = Field("", description="Base path of the model.")
    epoch: int = Field(0, description="Epoch number of the model.")
    save_ckpt_during: bool = Field(True, description="Whether to save checkpoints during training.")
    pretrained_model_name_or_path: str = Field(
        "",
        description="Path to pretrained model or model identifier from huggingface.co/models."
    )
    revision: Optional[str] = Field(
        None,
        description="Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be float32 precision."
    )
    instance_data_dir: str = Field(
        "",
        description="A folder containing the training data of instance images."
    )
    class_data_dir: Optional[str] = Field(
        "",
        description="A folder containing the training data of class images."
    )
    instance_prompt: str = Field(
        "",
        description="The prompt with identifier specifying the instance"
    )
    class_prompt: Optional[str] = Field(
        "",
        description="The prompt to specify images in the same class as provided instance images."
    )
    validation_prompt: str = Field(
        "",
        description="The prompt for validation."
    )
    with_prior_preservation: bool = Field(
        False,
        description="Flag to add prior preservation loss."
    )
    prior_loss_weight: float = Field(
        1.0,
        description="The weight of prior preservation loss."
    )
    num_class_images: int = Field(
        100,
        description="Minimal class images for prior preservation loss. If there are not enough images already present in class_data_dir, additional images will be sampled with class_prompt."
    )
    seed: Optional[int] = Field(
        -1,
        description="A seed for reproducible training."
    )
    resolution: int = Field(
        512,
        description="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"
    )
    center_crop: bool = Field(
        False,
        description="Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping."
    )
    train_text_encoder: bool = Field(
        False,
        description="Whether to train the text encoder. If set, the text encoder should be float32 precision."
    )
    train_batch_size: int = Field(
        4,
        description="Batch size (per device) for the training dataloader."
    )
    sample_batch_size: int = Field(
        4,
        description="Batch size (per device) for sampling images."
    )
    num_train_epochs: int = Field(
        100,
        description="Number of training epochs."
    )
    save_embedding_every: int = Field(
        500,
        description="Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via --resume_from_checkpoint. In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference. Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
    )
    checkpoints_total_limit: Optional[int] = Field(
        None,
        description="Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`. See Accelerator::save_state for more details."
    )
    snapshot: Optional[str] = Field(
        "",
        description="Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `latest` to automatically select the last available checkpoint."
    )
    gradient_accumulation_steps: int = Field(
        1,
        description="Number of updates steps to accumulate before performing a backward/update pass."
    )
    gradient_checkpointing: bool = Field(
        False,
        description="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass."
    )
    learning_rate: float = Field(
        5e-6,
        description="Initial learning rate (after the potential warmup period) to use."
    )
    scale_lr: bool = Field(
        False,
        description="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size."
    )
    lr_scheduler: str = Field(
        "constant",
        description='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
    )
    lr_warmup_steps: int = Field(
        500,
        description="Number of steps for the warmup in the lr scheduler."
    )
    lr_num_cycles: int = Field(
        1,
        description="Number of hard resets of the lr in cosine_with_restarts scheduler."
    )
    lr_power: float = Field(
        1.0,
        description="Power factor of the polynomial scheduler."
    )
    optimizer: str = Field("adamw", description="The optimizer to use.")
    adam_beta1: float = Field(
        0.9,
        description="The beta1 parameter for the Adam optimizer."
    )
    adam_beta2: float = Field(
        0.999,
        description="The beta2 parameter for the Adam optimizer."
    )
    adam_weight_decay: float = Field(
        1e-2,
        description="Weight decay to use."
    )
    adam_epsilon: float = Field(
        1e-08,
        description="Epsilon value for the Adam optimizer."
    )
    max_grad_norm: float = Field(
        1.0,
        description="Max gradient norm."
    )
    allow_tf32: bool = Field(
        False,
        description="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
    )
    num_validation_images: int = Field(
        4,
        description="Number of images that should be generated during validation with `validation_prompt`."
    )
    save_preview_every: int = Field(
        100,
        description="Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images` and logging the images."
    )
    mixed_precision: Optional[str] = Field(
        None,
        description="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    )
    prior_generation_precision: Optional[str] = Field(
        None,
        description="Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU. Default to fp16 if a GPU is available else fp32."
    )
    local_rank: int = Field(
        -1,
        description="For distributed training: local_rank"
    )
    attention: str = Field(
        "xformers",
        description="The memory attention to use."
    )
    gradient_set_to_none: bool = Field(
        False,
        description="Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain behaviors, so disable this argument if it causes any problems. More info: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
    )
    offset_noise: float = Field(
        0.0,
        description="Fine-tuning against a modified noise. See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information."
    )
    pre_compute_text_embeddings: bool = Field(
        False,
        description="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`."
    )
    tokenizer_max_length: Optional[int] = Field(
        None,
        description="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length."
    )
    text_encoder_use_attention_mask: bool = Field(
        False,
        description="Whether to use attention mask for the text encoder"
    )
    skip_save_text_encoder: bool = Field(
        False,
        description="Set to not save text encoder"
    )
    validation_images: Optional[List[str]] = Field(
        None,
        description="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution."
    )
    class_labels_conditioning: Optional[str] = Field(
        None,
        description="The optional `class_label` conditioning to pass to the unet, available values are timesteps."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

