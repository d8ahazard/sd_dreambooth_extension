import gradio as gr

from dreambooth.utils.ui_utils import ElementManager

manager = ElementManager()

GRADIENT_ACCUMULATION_STEPS = None
TRAIN_BATCH_SIZE = None
CAPTION_COLUMN = None
DATASET_CONFIG_NAME = None
DATASET_NAME = None
IMAGE_COLUMN = None
PROPORTION_EMPTY_PROMPTS = None
TRAIN_DATA_DIR = None
CENTER_CROP = None
RANDOM_FLIP = None
RESOLUTION = None
CHECKPOINTING_STEPS = None
MAX_TRAIN_SAMPLES = None
MAX_TRAIN_STEPS = None
NUM_TRAIN_EPOCHS = None
VALIDATION_EPOCHS = None
LEARNING_RATE = None
LR_SCHEDULER = None
LR_WARMUP_STEPS = None
SCALE_LR = None
LOGGING_DIR = None
REPORT_TO = None
PRETRAINED_VAE_MODEL_NAME_OR_PATH = None
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
NOISE_OFFSET = None
PREDICTION_TYPE = None
SEED = None
SNR_GAMMA = None
TIMESTEP_BIAS_BEGIN = None
TIMESTEP_BIAS_END = None
TIMESTEP_BIAS_MULTIPLIER = None
TIMESTEP_BIAS_PORTION = None
TIMESTEP_BIAS_STRATEGY = None
USE_8BIT_ADAM = None
USE_EMA = None
CHECKPOINTS_TOTAL_LIMIT = None
HUB_MODEL_ID = None
HUB_TOKEN = None
PUSH_TO_HUB = None
NUM_VALIDATION_IMAGES = None
VALIDATION_PROMPT = None


def render():
    global GRADIENT_ACCUMULATION_STEPS
    global TRAIN_BATCH_SIZE
    global CAPTION_COLUMN
    global DATASET_CONFIG_NAME
    global DATASET_NAME
    global IMAGE_COLUMN
    global PROPORTION_EMPTY_PROMPTS
    global TRAIN_DATA_DIR
    global CENTER_CROP
    global RANDOM_FLIP
    global RESOLUTION
    global CHECKPOINTING_STEPS
    global MAX_TRAIN_SAMPLES
    global MAX_TRAIN_STEPS
    global NUM_TRAIN_EPOCHS
    global VALIDATION_EPOCHS
    global LEARNING_RATE
    global LR_SCHEDULER
    global LR_WARMUP_STEPS
    global SCALE_LR
    global LOGGING_DIR
    global REPORT_TO
    global PRETRAINED_VAE_MODEL_NAME_OR_PATH
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
    global NOISE_OFFSET
    global PREDICTION_TYPE
    global SEED
    global SNR_GAMMA
    global TIMESTEP_BIAS_BEGIN
    global TIMESTEP_BIAS_END
    global TIMESTEP_BIAS_MULTIPLIER
    global TIMESTEP_BIAS_PORTION
    global TIMESTEP_BIAS_STRATEGY
    global USE_8BIT_ADAM
    global USE_EMA
    global CHECKPOINTS_TOTAL_LIMIT
    global HUB_MODEL_ID
    global HUB_TOKEN
    global PUSH_TO_HUB
    global NUM_VALIDATION_IMAGES
    global VALIDATION_PROMPT
    with gr.Accordion(open=False, label="Batching") as BATCHING_ACCORDION:
        GRADIENT_ACCUMULATION_STEPS = gr.Slider(interactive=True, label='Gradient Accumulation Steps', value=1, visible=True, step=1, minimum=0, maximum=100)
        TRAIN_BATCH_SIZE = gr.Slider(interactive=True, label='Train Batch Size', value=4, visible=True, step=1, minimum=1, maximum=1000)
    with gr.Accordion(open=False, label="Dataset") as DATASET_ACCORDION:
        CAPTION_COLUMN = gr.Textbox(interactive=True, label='Caption Column', value='text', visible=True)
        DATASET_CONFIG_NAME = gr.Textbox(interactive=True, label='Dataset Config Name', value='None', visible=True)
        DATASET_NAME = gr.Textbox(interactive=True, label='Dataset Name', value='None', visible=True)
        IMAGE_COLUMN = gr.Textbox(interactive=True, label='Image Column', value='image', visible=True)
        PROPORTION_EMPTY_PROMPTS = gr.Slider(interactive=True, label='Proportion Empty Prompts', value=0, visible=True, step=0.01, minimum=0, maximum=1)
        TRAIN_DATA_DIR = gr.Textbox(interactive=True, label='Train Data Dir', value='None', visible=True)
    with gr.Accordion(open=False, label="Image Processing") as IMAGE_PROCESSING_ACCORDION:
        CENTER_CROP = gr.Checkbox(interactive=True, label='Center Crop', value=False, visible=True)
        RANDOM_FLIP = gr.Checkbox(interactive=True, label='Random Flip', value=False, visible=True)
        RESOLUTION = gr.Slider(interactive=True, label='Resolution', value=1024, visible=True, step=64, minimum=256, maximum=4096)
    with gr.Accordion(open=False, label="Intervals") as INTERVALS_ACCORDION:
        CHECKPOINTING_STEPS = gr.Slider(interactive=True, label='Checkpointing Steps', value=500, visible=True, step=1, minimum=0, maximum=100000)
        MAX_TRAIN_SAMPLES = gr.Textbox(interactive=True, label='Max Train Samples', value='None', visible=True)
        MAX_TRAIN_STEPS = gr.Textbox(interactive=True, label='Max Train Steps', value='None', visible=False)
        NUM_TRAIN_EPOCHS = gr.Slider(interactive=True, label='Num Train Epochs', value=1, visible=True, step=1, minimum=0, maximum=100000)
        VALIDATION_EPOCHS = gr.Slider(interactive=True, label='Validation Epochs', value=1, visible=True, step=1, minimum=0, maximum=100)
    with gr.Accordion(open=False, label="Learning Rate") as LEARNING_RATE_ACCORDION:
        LEARNING_RATE = gr.Slider(interactive=True, label='Learning Rate', value=5e-06, visible=True, step=0.01, minimum=0, maximum=1)
        LR_SCHEDULER = gr.Dropdown(interactive=True, label='Lr Scheduler', choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'], value='constant', visible=True)
        LR_WARMUP_STEPS = gr.Slider(interactive=True, label='Lr Warmup Steps', value=500, visible=True, step=1, minimum=0, maximum=100000)
        SCALE_LR = gr.Checkbox(interactive=True, label='Scale Lr', value=False, visible=True)
    with gr.Accordion(open=False, label="Logging") as LOGGING_ACCORDION:
        LOGGING_DIR = gr.Textbox(interactive=True, label='Logging Dir', value='logs', visible=False)
        REPORT_TO = gr.Dropdown(interactive=True, label='Report To', choices=['all', 'tensorboard', 'wandb'], value='tensorboard', visible=True)
    with gr.Accordion(open=False, label="Model") as MODEL_ACCORDION:
        PRETRAINED_VAE_MODEL_NAME_OR_PATH = gr.Textbox(interactive=True, label='Pretrained Vae Model Name Or Path', value='None', visible=True)
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
        NOISE_OFFSET = gr.Slider(interactive=True, label='Noise Offset', value=0, visible=True, step=0.01, minimum=0, maximum=1)
        PREDICTION_TYPE = gr.Dropdown(interactive=True, label='Prediction Type', choices=['epsilon', 'v_prediction'], value='None', visible=True)
        SEED = gr.Textbox(interactive=True, label='Seed', value='None', visible=True)
        SNR_GAMMA = gr.Textbox(interactive=True, label='Snr Gamma', value='None', visible=True)
        TIMESTEP_BIAS_BEGIN = gr.Slider(interactive=True, label='Timestep Bias Begin', value=0, visible=True, step=1, minimum=0, maximum=100000)
        TIMESTEP_BIAS_END = gr.Slider(interactive=True, label='Timestep Bias End', value=1000, visible=True, step=1, minimum=0, maximum=100000)
        TIMESTEP_BIAS_MULTIPLIER = gr.Slider(interactive=True, label='Timestep Bias Multiplier', value=1.0, visible=True, step=0.01, minimum=0, maximum=1)
        TIMESTEP_BIAS_PORTION = gr.Slider(interactive=True, label='Timestep Bias Portion', value=0.25, visible=True, step=0.01, minimum=0, maximum=1)
        TIMESTEP_BIAS_STRATEGY = gr.Dropdown(interactive=True, label='Timestep Bias Strategy', choices=['earlier', 'later', 'range', 'none'], value='none', visible=True)
        USE_8BIT_ADAM = gr.Checkbox(interactive=True, label='Use 8Bit Adam', value=True, visible=True)
        USE_EMA = gr.Checkbox(interactive=True, label='Use Ema', value=False, visible=True)
    with gr.Accordion(open=False, label="Saving") as SAVING_ACCORDION:
        CHECKPOINTS_TOTAL_LIMIT = gr.Textbox(interactive=True, label='Checkpoints Total Limit', value='3', visible=True)
        HUB_MODEL_ID = gr.Textbox(interactive=True, label='Hub Model Id', value='None', visible=False)
        HUB_TOKEN = gr.Textbox(interactive=True, label='Hub Token', value='None', visible=False)
        PUSH_TO_HUB = gr.Checkbox(interactive=True, label='Push To Hub', value=False, visible=False)
    with gr.Accordion(open=False, label="Validation") as VALIDATION_ACCORDION:
        NUM_VALIDATION_IMAGES = gr.Slider(interactive=True, label='Num Validation Images', value=4, visible=True, step=1, minimum=0, maximum=100)
        VALIDATION_PROMPT = gr.Textbox(interactive=True, label='Validation Prompt', value='None', visible=True)
    manager.register_db_component("train_text_to_image_sdxl", GRADIENT_ACCUMULATION_STEPS, "gradient_accumulation_steps", True, "Number of updates steps to accumulate before performing a backward/update pass.")
    manager.register_db_component("train_text_to_image_sdxl", TRAIN_BATCH_SIZE, "train_batch_size", False, "Batch size (per device) for the training dataloader.")
    manager.register_db_component("train_text_to_image_sdxl", CAPTION_COLUMN, "caption_column", False, "The column of the dataset containing a caption or a list of captions.")
    manager.register_db_component("train_text_to_image_sdxl", DATASET_CONFIG_NAME, "dataset_config_name", True, "The config of the Dataset, leave as None if there\'s only one config.")
    manager.register_db_component("train_text_to_image_sdxl", DATASET_NAME, "dataset_name", False, "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private, dataset). It can also be a path pointing to a local copy of a dataset in your filesystem, or to a folder containing files that 🤗 Datasets can understand.")
    manager.register_db_component("train_text_to_image_sdxl", IMAGE_COLUMN, "image_column", False, "The column of the dataset containing an image.")
    manager.register_db_component("train_text_to_image_sdxl", PROPORTION_EMPTY_PROMPTS, "proportion_empty_prompts", True, "Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).")
    manager.register_db_component("train_text_to_image_sdxl", TRAIN_DATA_DIR, "train_data_dir", False, "A folder containing the training data. Folder contents must follow the structure described in https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file must exist to provide the captions for the images. Ignored if `dataset_name` is specified.")
    manager.register_db_component("train_text_to_image_sdxl", CENTER_CROP, "center_crop", True, "Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.")
    manager.register_db_component("train_text_to_image_sdxl", RANDOM_FLIP, "random_flip", True, "whether to randomly flip images horizontally")
    manager.register_db_component("train_text_to_image_sdxl", RESOLUTION, "resolution", False, "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution")
    manager.register_db_component("train_text_to_image_sdxl", CHECKPOINTING_STEPS, "checkpointing_steps", False, "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final checkpoints in case they are better than the last checkpoint, and are also suitable for resuming training using `--resume_from_checkpoint`.")
    manager.register_db_component("train_text_to_image_sdxl", MAX_TRAIN_SAMPLES, "max_train_samples", True, "For debugging purposes or quicker training, truncate the number of training examples to this value if set.")
    manager.register_db_component("train_text_to_image_sdxl", MAX_TRAIN_STEPS, "max_train_steps", False, "Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    manager.register_db_component("train_text_to_image_sdxl", NUM_TRAIN_EPOCHS, "num_train_epochs", False, "")
    manager.register_db_component("train_text_to_image_sdxl", VALIDATION_EPOCHS, "validation_epochs", False, "Run fine-tuning validation every X epochs. The validation process consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images`.")
    manager.register_db_component("train_text_to_image_sdxl", LEARNING_RATE, "learning_rate", False, "Initial learning rate (after the potential warmup period) to use.")
    manager.register_db_component("train_text_to_image_sdxl", LR_SCHEDULER, "lr_scheduler", True, "The scheduler type to use. Choose between [\"linear\", \"cosine\", \"cosine_with_restarts\", \"polynomial\", \"constant\", \"constant_with_warmup\"]")
    manager.register_db_component("train_text_to_image_sdxl", LR_WARMUP_STEPS, "lr_warmup_steps", True, "Number of steps for the warmup in the lr scheduler.")
    manager.register_db_component("train_text_to_image_sdxl", SCALE_LR, "scale_lr", True, "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    manager.register_db_component("train_text_to_image_sdxl", LOGGING_DIR, "logging_dir", False, "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    manager.register_db_component("train_text_to_image_sdxl", REPORT_TO, "report_to", False, "The integration to report the results and logs to. Supported platforms are `\"tensorboard\"` (default), `\"wandb\"` and `\"comet_ml\"`. Use `\"all\"` to report to all integrations.")
    manager.register_db_component("train_text_to_image_sdxl", PRETRAINED_VAE_MODEL_NAME_OR_PATH, "pretrained_vae_model_name_or_path", True, "Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.")
    manager.register_db_component("train_text_to_image_sdxl", ADAM_BETA1, "adam_beta1", True, "The beta1 parameter for the Adam optimizer.")
    manager.register_db_component("train_text_to_image_sdxl", ADAM_BETA2, "adam_beta2", True, "The beta2 parameter for the Adam optimizer.")
    manager.register_db_component("train_text_to_image_sdxl", ADAM_EPSILON, "adam_epsilon", True, "Epsilon value for the Adam optimizer")
    manager.register_db_component("train_text_to_image_sdxl", ADAM_WEIGHT_DECAY, "adam_weight_decay", True, "Weight decay to use.")
    manager.register_db_component("train_text_to_image_sdxl", MAX_GRAD_NORM, "max_grad_norm", True, "Max gradient norm.")
    manager.register_db_component("train_text_to_image_sdxl", ALLOW_TF32, "allow_tf32", True, "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    manager.register_db_component("train_text_to_image_sdxl", DATALOADER_NUM_WORKERS, "dataloader_num_workers", True, "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    manager.register_db_component("train_text_to_image_sdxl", ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION, "enable_xformers_memory_efficient_attention", False, "Whether or not to use xformers.")
    manager.register_db_component("train_text_to_image_sdxl", GRADIENT_CHECKPOINTING, "gradient_checkpointing", True, "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    manager.register_db_component("train_text_to_image_sdxl", LOCAL_RANK, "local_rank", True, "For distributed training: local_rank")
    manager.register_db_component("train_text_to_image_sdxl", MIXED_PRECISION, "mixed_precision", True, "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.")
    manager.register_db_component("train_text_to_image_sdxl", NOISE_OFFSET, "noise_offset", True, "The scale of noise offset.")
    manager.register_db_component("train_text_to_image_sdxl", PREDICTION_TYPE, "prediction_type", True, "The prediction_type that shall be used for training. Choose between \'epsilon\' or \'v_prediction\' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.")
    manager.register_db_component("train_text_to_image_sdxl", SEED, "seed", True, "A seed for reproducible training.")
    manager.register_db_component("train_text_to_image_sdxl", SNR_GAMMA, "snr_gamma", True, "SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556.")
    manager.register_db_component("train_text_to_image_sdxl", TIMESTEP_BIAS_BEGIN, "timestep_bias_begin", True, "When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias. Defaults to zero, which equates to having no specific bias.")
    manager.register_db_component("train_text_to_image_sdxl", TIMESTEP_BIAS_END, "timestep_bias_end", True, "When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias. Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on.")
    manager.register_db_component("train_text_to_image_sdxl", TIMESTEP_BIAS_MULTIPLIER, "timestep_bias_multiplier", True, "The multiplier for the bias. Defaults to 1.0, which means no bias is applied. A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it.")
    manager.register_db_component("train_text_to_image_sdxl", TIMESTEP_BIAS_PORTION, "timestep_bias_portion", True, "The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased. A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines whether the biased portions are in the earlier or later timesteps.")
    manager.register_db_component("train_text_to_image_sdxl", TIMESTEP_BIAS_STRATEGY, "timestep_bias_strategy", True, "The timestep bias strategy, which may help direct the model toward learning low or high frequency details. Choices: [\'earlier\', \'later\', \'range\', \'none\']. The default is \'none\', which means no bias is applied, and training proceeds normally. The value of \'later\' will increase the frequency of the model\'s final training timesteps.")
    manager.register_db_component("train_text_to_image_sdxl", USE_8BIT_ADAM, "use_8bit_adam", True, "Whether or not to use 8-bit Adam from bitsandbytes.")
    manager.register_db_component("train_text_to_image_sdxl", USE_EMA, "use_ema", True, "Whether to use EMA model.")
    manager.register_db_component("train_text_to_image_sdxl", CHECKPOINTS_TOTAL_LIMIT, "checkpoints_total_limit", True, "Max number of checkpoints to store.")
    manager.register_db_component("train_text_to_image_sdxl", HUB_MODEL_ID, "hub_model_id", False, "The name of the repository to keep in sync with the local `output_dir`.")
    manager.register_db_component("train_text_to_image_sdxl", HUB_TOKEN, "hub_token", False, "The token to use to push to the Model Hub.")
    manager.register_db_component("train_text_to_image_sdxl", PUSH_TO_HUB, "push_to_hub", False, "Whether or not to push the model to the Hub.")
    manager.register_db_component("train_text_to_image_sdxl", NUM_VALIDATION_IMAGES, "num_validation_images", True, "Number of images that should be generated during validation with `validation_prompt`.")
    manager.register_db_component("train_text_to_image_sdxl", VALIDATION_PROMPT, "validation_prompt", True, "A prompt that is used during validation to verify that the model is learning.")