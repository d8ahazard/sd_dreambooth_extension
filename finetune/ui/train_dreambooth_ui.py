import gradio as gr

from dreambooth.utils.ui_utils import ElementManager

manager = ElementManager()

GRADIENT_ACCUMULATION_STEPS = None
SAMPLE_BATCH_SIZE = None
TRAIN_BATCH_SIZE = None
CLASS_DATA_DIR = None
CLASS_LABELS_CONDITIONING = None
CLASS_PROMPT = None
INSTANCE_DATA_DIR = None
INSTANCE_PROMPT = None
NUM_CLASS_IMAGES = None
PRIOR_LOSS_WEIGHT = None
WITH_PRIOR_PRESERVATION = None
CENTER_CROP = None
RESOLUTION = None
CHECKPOINTING_STEPS = None
MAX_TRAIN_STEPS = None
NUM_TRAIN_EPOCHS = None
VALIDATION_STEPS = None
LEARNING_RATE = None
LR_NUM_CYCLES = None
LR_POWER = None
LR_SCHEDULER = None
LR_WARMUP_STEPS = None
SCALE_LR = None
LOGGING_DIR = None
REPORT_TO = None
ADAM_BETA1 = None
ADAM_BETA2 = None
ADAM_EPSILON = None
ADAM_WEIGHT_DECAY = None
MAX_GRAD_NORM = None
ALLOW_TF32 = None
DATALOADER_NUM_WORKERS = None
ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION = None
GRADIENT_CHECKPOINTING = None
LOCAL_RANK = None
MIXED_PRECISION = None
OFFSET_NOISE = None
PRE_COMPUTE_TEXT_EMBEDDINGS = None
PRIOR_GENERATION_PRECISION = None
SEED = None
SET_GRADS_TO_NONE = None
SNR_GAMMA = None
TEXT_ENCODER_USE_ATTENTION_MASK = None
TOKENIZER_MAX_LENGTH = None
TRAIN_TEXT_ENCODER = None
USE_8BIT_ADAM = None
CHECKPOINTS_TOTAL_LIMIT = None
HUB_MODEL_ID = None
HUB_TOKEN = None
PUSH_TO_HUB = None
SKIP_SAVE_TEXT_ENCODER = None
NUM_VALIDATION_IMAGES = None
VALIDATION_IMAGES = None
VALIDATION_PROMPT = None
VALIDATION_SCHEDULER = None


def render():
    global GRADIENT_ACCUMULATION_STEPS
    global SAMPLE_BATCH_SIZE
    global TRAIN_BATCH_SIZE
    global CLASS_DATA_DIR
    global CLASS_LABELS_CONDITIONING
    global CLASS_PROMPT
    global INSTANCE_DATA_DIR
    global INSTANCE_PROMPT
    global NUM_CLASS_IMAGES
    global PRIOR_LOSS_WEIGHT
    global WITH_PRIOR_PRESERVATION
    global CENTER_CROP
    global RESOLUTION
    global CHECKPOINTING_STEPS
    global MAX_TRAIN_STEPS
    global NUM_TRAIN_EPOCHS
    global VALIDATION_STEPS
    global LEARNING_RATE
    global LR_NUM_CYCLES
    global LR_POWER
    global LR_SCHEDULER
    global LR_WARMUP_STEPS
    global SCALE_LR
    global LOGGING_DIR
    global REPORT_TO
    global ADAM_BETA1
    global ADAM_BETA2
    global ADAM_EPSILON
    global ADAM_WEIGHT_DECAY
    global MAX_GRAD_NORM
    global ALLOW_TF32
    global DATALOADER_NUM_WORKERS
    global ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION
    global GRADIENT_CHECKPOINTING
    global LOCAL_RANK
    global MIXED_PRECISION
    global OFFSET_NOISE
    global PRE_COMPUTE_TEXT_EMBEDDINGS
    global PRIOR_GENERATION_PRECISION
    global SEED
    global SET_GRADS_TO_NONE
    global SNR_GAMMA
    global TEXT_ENCODER_USE_ATTENTION_MASK
    global TOKENIZER_MAX_LENGTH
    global TRAIN_TEXT_ENCODER
    global USE_8BIT_ADAM
    global CHECKPOINTS_TOTAL_LIMIT
    global HUB_MODEL_ID
    global HUB_TOKEN
    global PUSH_TO_HUB
    global SKIP_SAVE_TEXT_ENCODER
    global NUM_VALIDATION_IMAGES
    global VALIDATION_IMAGES
    global VALIDATION_PROMPT
    global VALIDATION_SCHEDULER
    with gr.Accordion(open=False, label="Batching") as BATCHING_ACCORDION:
        GRADIENT_ACCUMULATION_STEPS = gr.Slider(interactive=True, label='Gradient Accumulation Steps', value=1, visible=True, step=1, minimum=0, maximum=100)
        SAMPLE_BATCH_SIZE = gr.Slider(interactive=True, label='Sample Batch Size', value=4, visible=True, step=1, minimum=1, maximum=1000)
        TRAIN_BATCH_SIZE = gr.Slider(interactive=True, label='Train Batch Size', value=4, visible=True, step=1, minimum=1, maximum=1000)
    with gr.Accordion(open=False, label="Dataset") as DATASET_ACCORDION:
        CLASS_DATA_DIR = gr.Textbox(interactive=True, label='Class Data Dir', value='None', visible=True)
        CLASS_LABELS_CONDITIONING = gr.Textbox(interactive=True, label='Class Labels Conditioning', value='None', visible=True)
        CLASS_PROMPT = gr.Textbox(interactive=True, label='Class Prompt', value='None', visible=True)
        INSTANCE_DATA_DIR = gr.Textbox(interactive=True, label='Instance Data Dir', value='None', visible=True)
        INSTANCE_PROMPT = gr.Textbox(interactive=True, label='Instance Prompt', value='None', visible=True)
        NUM_CLASS_IMAGES = gr.Slider(interactive=True, label='Num Class Images', value=100, visible=True, step=1, minimum=0, maximum=100000)
        PRIOR_LOSS_WEIGHT = gr.Slider(interactive=True, label='Prior Loss Weight', value=1.0, visible=True, step=0.1, minimum=0, maximum=1)
        WITH_PRIOR_PRESERVATION = gr.Checkbox(interactive=True, label='With Prior Preservation', value=False, visible=True)
    with gr.Accordion(open=False, label="Image Processing") as IMAGE_PROCESSING_ACCORDION:
        CENTER_CROP = gr.Checkbox(interactive=True, label='Center Crop', value=False, visible=True)
        RESOLUTION = gr.Slider(interactive=True, label='Resolution', value=1024, visible=True, step=64, minimum=256, maximum=4096)
    with gr.Accordion(open=False, label="Intervals") as INTERVALS_ACCORDION:
        CHECKPOINTING_STEPS = gr.Slider(interactive=True, label='Checkpointing Steps', value=500, visible=True, step=1, minimum=0, maximum=100000)
        MAX_TRAIN_STEPS = gr.Textbox(interactive=True, label='Max Train Steps', value='None', visible=False)
        NUM_TRAIN_EPOCHS = gr.Slider(interactive=True, label='Num Train Epochs', value=1, visible=True, step=1, minimum=0, maximum=100000)
        VALIDATION_STEPS = gr.Slider(interactive=True, label='Validation Steps', value=100, visible=True, step=1, minimum=0, maximum=100)
    with gr.Accordion(open=False, label="Learning Rate") as LEARNING_RATE_ACCORDION:
        LEARNING_RATE = gr.Slider(interactive=True, label='Learning Rate', value=5e-06, visible=True, step=0.01, minimum=0, maximum=1)
        LR_NUM_CYCLES = gr.Slider(interactive=True, label='Lr Num Cycles', value=1, visible=True, step=1, minimum=0, maximum=100000)
        LR_POWER = gr.Slider(interactive=True, label='Lr Power', value=1.0, visible=True, step=0.1, minimum=0, maximum=1)
        LR_SCHEDULER = gr.Dropdown(interactive=True, label='Lr Scheduler', choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'], value='constant', visible=True)
        LR_WARMUP_STEPS = gr.Slider(interactive=True, label='Lr Warmup Steps', value=500, visible=True, step=1, minimum=0, maximum=100000)
        SCALE_LR = gr.Checkbox(interactive=True, label='Scale Lr', value=False, visible=True)
    with gr.Accordion(open=False, label="Logging") as LOGGING_ACCORDION:
        LOGGING_DIR = gr.Textbox(interactive=True, label='Logging Dir', value='logs', visible=False)
        REPORT_TO = gr.Dropdown(interactive=True, label='Report To', choices=['all', 'tensorboard', 'wandb'], value='tensorboard', visible=True)
    with gr.Accordion(open=False, label="Optimizer") as OPTIMIZER_ACCORDION:
        ADAM_BETA1 = gr.Slider(interactive=True, label='Adam Beta1', value=0.9, visible=True, step=0.01, minimum=0, maximum=1)
        ADAM_BETA2 = gr.Slider(interactive=True, label='Adam Beta2', value=0.999, visible=True, step=0.01, minimum=0, maximum=1)
        ADAM_EPSILON = gr.Slider(interactive=True, label='Adam Epsilon', value=1e-08, visible=True, step=0.01, minimum=0, maximum=1)
        ADAM_WEIGHT_DECAY = gr.Slider(interactive=True, label='Adam Weight Decay', value=0.01, visible=True, step=0.01, minimum=0, maximum=1)
        MAX_GRAD_NORM = gr.Slider(interactive=True, label='Max Grad Norm', value=1.0, visible=True, step=0.1, minimum=0, maximum=1)
    with gr.Accordion(open=False, label="Performance") as PERFORMANCE_ACCORDION:
        ALLOW_TF32 = gr.Checkbox(interactive=True, label='Allow Tf32', value=True, visible=True)
        DATALOADER_NUM_WORKERS = gr.Slider(interactive=True, label='Dataloader Num Workers', value=1, visible=True, step=1, minimum=0, maximum=100)
        ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION = gr.Checkbox(interactive=True, label='Enable Xformers Memory Efficient Attention', value=True, visible=True)
        GRADIENT_CHECKPOINTING = gr.Checkbox(interactive=True, label='Gradient Checkpointing', value=True, visible=True)
        LOCAL_RANK = gr.Slider(interactive=True, label='Local Rank', value=-1, visible=True, step=1, minimum=0, maximum=100)
        MIXED_PRECISION = gr.Dropdown(interactive=True, label='Mixed Precision', choices=['no', 'fp16', 'bf16'], value='bf16', visible=True)
        OFFSET_NOISE = gr.Checkbox(interactive=True, label='Offset Noise', value=False, visible=True)
        PRE_COMPUTE_TEXT_EMBEDDINGS = gr.Checkbox(interactive=True, label='Pre Compute Text Embeddings', value=False, visible=True)
        PRIOR_GENERATION_PRECISION = gr.Dropdown(interactive=True, label='Prior Generation Precision', choices=['no', 'fp32', 'fp16', 'bf16'], value='bf16', visible=True)
        SEED = gr.Textbox(interactive=True, label='Seed', value='None', visible=True)
        SET_GRADS_TO_NONE = gr.Checkbox(interactive=True, label='Set Grads To None', value=False, visible=True)
        SNR_GAMMA = gr.Textbox(interactive=True, label='Snr Gamma', value='None', visible=True)
        TEXT_ENCODER_USE_ATTENTION_MASK = gr.Checkbox(interactive=True, label='Text Encoder Use Attention Mask', value=False, visible=True)
        TOKENIZER_MAX_LENGTH = gr.Textbox(interactive=True, label='Tokenizer Max Length', value='None', visible=True)
        TRAIN_TEXT_ENCODER = gr.Checkbox(interactive=True, label='Train Text Encoder', value=False, visible=True)
        USE_8BIT_ADAM = gr.Checkbox(interactive=True, label='Use 8Bit Adam', value=True, visible=True)
    with gr.Accordion(open=False, label="Saving") as SAVING_ACCORDION:
        CHECKPOINTS_TOTAL_LIMIT = gr.Textbox(interactive=True, label='Checkpoints Total Limit', value='3', visible=True)
        HUB_MODEL_ID = gr.Textbox(interactive=True, label='Hub Model Id', value='None', visible=False)
        HUB_TOKEN = gr.Textbox(interactive=True, label='Hub Token', value='None', visible=False)
        PUSH_TO_HUB = gr.Checkbox(interactive=True, label='Push To Hub', value=False, visible=False)
        SKIP_SAVE_TEXT_ENCODER = gr.Checkbox(interactive=True, label='Skip Save Text Encoder', value=False, visible=True)
    with gr.Accordion(open=False, label="Validation") as VALIDATION_ACCORDION:
        NUM_VALIDATION_IMAGES = gr.Slider(interactive=True, label='Num Validation Images', value=4, visible=True, step=1, minimum=0, maximum=100)
        VALIDATION_IMAGES = gr.Textbox(interactive=True, label='Validation Images', value='None', visible=True)
        VALIDATION_PROMPT = gr.Textbox(interactive=True, label='Validation Prompt', value='None', visible=True)
        VALIDATION_SCHEDULER = gr.Dropdown(interactive=True, label='Validation Scheduler', choices=['DPMSolverMultistepScheduler', 'DDPMScheduler'], value='DPMSolverMultistepScheduler', visible=True)
    manager.register_db_component("train_dreambooth", GRADIENT_ACCUMULATION_STEPS, "gradient_accumulation_steps", True, "Number of updates steps to accumulate before performing a backward/update pass.")
    manager.register_db_component("train_dreambooth", SAMPLE_BATCH_SIZE, "sample_batch_size", True, "Batch size (per device) for sampling images.")
    manager.register_db_component("train_dreambooth", TRAIN_BATCH_SIZE, "train_batch_size", False, "Batch size (per device) for the training dataloader.")
    manager.register_db_component("train_dreambooth", CLASS_DATA_DIR, "class_data_dir", False, "A folder containing the training data of class images.")
    manager.register_db_component("train_dreambooth", CLASS_LABELS_CONDITIONING, "class_labels_conditioning", False, "The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.")
    manager.register_db_component("train_dreambooth", CLASS_PROMPT, "class_prompt", False, "The prompt to specify images in the same class as provided instance images.")
    manager.register_db_component("train_dreambooth", INSTANCE_DATA_DIR, "instance_data_dir", False, "A folder containing the training data of instance images.")
    manager.register_db_component("train_dreambooth", INSTANCE_PROMPT, "instance_prompt", False, "The prompt with identifier specifying the instance")
    manager.register_db_component("train_dreambooth", NUM_CLASS_IMAGES, "num_class_images", False, "Minimal class images for prior preservation loss. If there are not enough images already present in class_data_dir, additional images will be sampled with class_prompt.")
    manager.register_db_component("train_dreambooth", PRIOR_LOSS_WEIGHT, "prior_loss_weight", True, "The weight of prior preservation loss.")
    manager.register_db_component("train_dreambooth", WITH_PRIOR_PRESERVATION, "with_prior_preservation", True, "Flag to add prior preservation loss.")
    manager.register_db_component("train_dreambooth", CENTER_CROP, "center_crop", True, "Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.")
    manager.register_db_component("train_dreambooth", RESOLUTION, "resolution", False, "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution")
    manager.register_db_component("train_dreambooth", CHECKPOINTING_STEPS, "checkpointing_steps", False, "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference.Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components.See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by stepinstructions.")
    manager.register_db_component("train_dreambooth", MAX_TRAIN_STEPS, "max_train_steps", False, "Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    manager.register_db_component("train_dreambooth", NUM_TRAIN_EPOCHS, "num_train_epochs", False, "")
    manager.register_db_component("train_dreambooth", VALIDATION_STEPS, "validation_steps", False, "Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images` and logging the images.")
    manager.register_db_component("train_dreambooth", LEARNING_RATE, "learning_rate", False, "Initial learning rate (after the potential warmup period) to use.")
    manager.register_db_component("train_dreambooth", LR_NUM_CYCLES, "lr_num_cycles", True, "Number of hard resets of the lr in cosine_with_restarts scheduler.")
    manager.register_db_component("train_dreambooth", LR_POWER, "lr_power", True, "Power factor of the polynomial scheduler.")
    manager.register_db_component("train_dreambooth", LR_SCHEDULER, "lr_scheduler", True, "The scheduler type to use. Choose between [\"linear\", \"cosine\", \"cosine_with_restarts\", \"polynomial\", \"constant\", \"constant_with_warmup\"]")
    manager.register_db_component("train_dreambooth", LR_WARMUP_STEPS, "lr_warmup_steps", True, "Number of steps for the warmup in the lr scheduler.")
    manager.register_db_component("train_dreambooth", SCALE_LR, "scale_lr", True, "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    manager.register_db_component("train_dreambooth", LOGGING_DIR, "logging_dir", False, "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    manager.register_db_component("train_dreambooth", REPORT_TO, "report_to", False, "The integration to report the results and logs to. Supported platforms are `\"tensorboard\"` (default), `\"wandb\"` and `\"comet_ml\"`. Use `\"all\"` to report to all integrations.")
    manager.register_db_component("train_dreambooth", ADAM_BETA1, "adam_beta1", True, "The beta1 parameter for the Adam optimizer.")
    manager.register_db_component("train_dreambooth", ADAM_BETA2, "adam_beta2", True, "The beta2 parameter for the Adam optimizer.")
    manager.register_db_component("train_dreambooth", ADAM_EPSILON, "adam_epsilon", True, "Epsilon value for the Adam optimizer")
    manager.register_db_component("train_dreambooth", ADAM_WEIGHT_DECAY, "adam_weight_decay", True, "Weight decay to use.")
    manager.register_db_component("train_dreambooth", MAX_GRAD_NORM, "max_grad_norm", True, "Max gradient norm.")
    manager.register_db_component("train_dreambooth", ALLOW_TF32, "allow_tf32", True, "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    manager.register_db_component("train_dreambooth", DATALOADER_NUM_WORKERS, "dataloader_num_workers", True, "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    manager.register_db_component("train_dreambooth", ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION, "enable_xformers_memory_efficient_attention", False, "Whether or not to use xformers.")
    manager.register_db_component("train_dreambooth", GRADIENT_CHECKPOINTING, "gradient_checkpointing", True, "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    manager.register_db_component("train_dreambooth", LOCAL_RANK, "local_rank", True, "For distributed training: local_rank")
    manager.register_db_component("train_dreambooth", MIXED_PRECISION, "mixed_precision", True, "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.")
    manager.register_db_component("train_dreambooth", OFFSET_NOISE, "offset_noise", True, "Fine-tuning against a modified noise See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information.")
    manager.register_db_component("train_dreambooth", PRE_COMPUTE_TEXT_EMBEDDINGS, "pre_compute_text_embeddings", True, "Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.")
    manager.register_db_component("train_dreambooth", PRIOR_GENERATION_PRECISION, "prior_generation_precision", True, "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32.")
    manager.register_db_component("train_dreambooth", SEED, "seed", True, "A seed for reproducible training.")
    manager.register_db_component("train_dreambooth", SET_GRADS_TO_NONE, "set_grads_to_none", True, "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain behaviors, so disable this argument if it causes any problems. More info: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html")
    manager.register_db_component("train_dreambooth", SNR_GAMMA, "snr_gamma", True, "SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556.")
    manager.register_db_component("train_dreambooth", TEXT_ENCODER_USE_ATTENTION_MASK, "text_encoder_use_attention_mask", True, "Whether to use attention mask for the text encoder")
    manager.register_db_component("train_dreambooth", TOKENIZER_MAX_LENGTH, "tokenizer_max_length", True, "The maximum length of the tokenizer. If not set, will default to the tokenizer\'s max length.")
    manager.register_db_component("train_dreambooth", TRAIN_TEXT_ENCODER, "train_text_encoder", True, "Whether to train the text encoder. If set, the text encoder should be float32 precision.")
    manager.register_db_component("train_dreambooth", USE_8BIT_ADAM, "use_8bit_adam", True, "Whether or not to use 8-bit Adam from bitsandbytes.")
    manager.register_db_component("train_dreambooth", CHECKPOINTS_TOTAL_LIMIT, "checkpoints_total_limit", True, "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`. See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state for more details")
    manager.register_db_component("train_dreambooth", HUB_MODEL_ID, "hub_model_id", False, "The name of the repository to keep in sync with the local `output_dir`.")
    manager.register_db_component("train_dreambooth", HUB_TOKEN, "hub_token", False, "The token to use to push to the Model Hub.")
    manager.register_db_component("train_dreambooth", PUSH_TO_HUB, "push_to_hub", False, "Whether or not to push the model to the Hub.")
    manager.register_db_component("train_dreambooth", SKIP_SAVE_TEXT_ENCODER, "skip_save_text_encoder", True, "Set to not save text encoder")
    manager.register_db_component("train_dreambooth", NUM_VALIDATION_IMAGES, "num_validation_images", True, "Number of images that should be generated during validation with `validation_prompt`.")
    manager.register_db_component("train_dreambooth", VALIDATION_IMAGES, "validation_images", True, "Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.")
    manager.register_db_component("train_dreambooth", VALIDATION_PROMPT, "validation_prompt", True, "A prompt that is used during validation to verify that the model is learning.")
    manager.register_db_component("train_dreambooth", VALIDATION_SCHEDULER, "validation_scheduler", True, "Select which scheduler to use for validation. DDPMScheduler is recommended for DeepFloyd IF.")