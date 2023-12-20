import gradio as gr

from dreambooth.utils.ui_utils import ElementManager

manager = ElementManager()

GRADIENT_ACCUMULATION_STEPS = None
TRAIN_BATCH_SIZE = None
CAPTION_COLUMN = None
CONDITIONING_IMAGE_COLUMN = None
CROPS_COORDS_TOP_LEFT_H = None
CROPS_COORDS_TOP_LEFT_W = None
DATASET_CONFIG_NAME = None
DATASET_NAME = None
IMAGE_COLUMN = None
PROPORTION_EMPTY_PROMPTS = None
TRAIN_DATA_DIR = None
RESOLUTION = None
CHECKPOINTING_STEPS = None
MAX_TRAIN_SAMPLES = None
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
TRACKER_PROJECT_NAME = None
CONTROLNET_MODEL_NAME_OR_PATH = None
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
MIXED_PRECISION = None
SEED = None
SET_GRADS_TO_NONE = None
USE_8BIT_ADAM = None
CHECKPOINTS_TOTAL_LIMIT = None
HUB_MODEL_ID = None
HUB_TOKEN = None
PUSH_TO_HUB = None
NUM_VALIDATION_IMAGES = None
VALIDATION_IMAGE = None
VALIDATION_PROMPT = None


def render():
    global GRADIENT_ACCUMULATION_STEPS
    global TRAIN_BATCH_SIZE
    global CAPTION_COLUMN
    global CONDITIONING_IMAGE_COLUMN
    global CROPS_COORDS_TOP_LEFT_H
    global CROPS_COORDS_TOP_LEFT_W
    global DATASET_CONFIG_NAME
    global DATASET_NAME
    global IMAGE_COLUMN
    global PROPORTION_EMPTY_PROMPTS
    global TRAIN_DATA_DIR
    global RESOLUTION
    global CHECKPOINTING_STEPS
    global MAX_TRAIN_SAMPLES
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
    global TRACKER_PROJECT_NAME
    global CONTROLNET_MODEL_NAME_OR_PATH
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
    global MIXED_PRECISION
    global SEED
    global SET_GRADS_TO_NONE
    global USE_8BIT_ADAM
    global CHECKPOINTS_TOTAL_LIMIT
    global HUB_MODEL_ID
    global HUB_TOKEN
    global PUSH_TO_HUB
    global NUM_VALIDATION_IMAGES
    global VALIDATION_IMAGE
    global VALIDATION_PROMPT
    with gr.Accordion(open=False, label="Batching") as BATCHING_ACCORDION:
        GRADIENT_ACCUMULATION_STEPS = gr.Slider(label='Gradient Accumulation Steps', value=1, visible=True, step=1, minimum=0, maximum=100)
        TRAIN_BATCH_SIZE = gr.Slider(label='Train Batch Size', value=4, visible=True, step=1, minimum=1, maximum=1000)
    with gr.Accordion(open=False, label="Dataset") as DATASET_ACCORDION:
        CAPTION_COLUMN = gr.Textbox(label='Caption Column', value='text', visible=True)
        CONDITIONING_IMAGE_COLUMN = gr.Textbox(label='Conditioning Image Column', value='conditioning_image', visible=True)
        CROPS_COORDS_TOP_LEFT_H = gr.Slider(label='Crops Coords Top Left H', value=0, visible=True, step=1, minimum=0, maximum=100000)
        CROPS_COORDS_TOP_LEFT_W = gr.Slider(label='Crops Coords Top Left W', value=0, visible=True, step=1, minimum=0, maximum=100000)
        DATASET_CONFIG_NAME = gr.Textbox(label='Dataset Config Name', value='None', visible=True)
        DATASET_NAME = gr.Textbox(label='Dataset Name', value='None', visible=True)
        IMAGE_COLUMN = gr.Textbox(label='Image Column', value='image', visible=True)
        PROPORTION_EMPTY_PROMPTS = gr.Slider(label='Proportion Empty Prompts', value=0, visible=True, step=0.01, minimum=0, maximum=1)
        TRAIN_DATA_DIR = gr.Textbox(label='Train Data Dir', value='None', visible=True)
    with gr.Accordion(open=False, label="Image Processing") as IMAGE_PROCESSING_ACCORDION:
        RESOLUTION = gr.Slider(label='Resolution', value=1024, visible=True, step=64, minimum=256, maximum=4096)
    with gr.Accordion(open=False, label="Intervals") as INTERVALS_ACCORDION:
        CHECKPOINTING_STEPS = gr.Slider(label='Checkpointing Steps', value=500, visible=True, step=1, minimum=0, maximum=100000)
        MAX_TRAIN_SAMPLES = gr.Textbox(label='Max Train Samples', value='None', visible=True)
        MAX_TRAIN_STEPS = gr.Textbox(label='Max Train Steps', value='None', visible=False)
        NUM_TRAIN_EPOCHS = gr.Slider(label='Num Train Epochs', value=1, visible=True, step=1, minimum=0, maximum=100000)
        VALIDATION_STEPS = gr.Slider(label='Validation Steps', value=100, visible=True, step=1, minimum=0, maximum=100)
    with gr.Accordion(open=False, label="Learning Rate") as LEARNING_RATE_ACCORDION:
        LEARNING_RATE = gr.Slider(label='Learning Rate', value=5e-06, visible=True, step=0.01, minimum=0, maximum=1)
        LR_NUM_CYCLES = gr.Slider(label='Lr Num Cycles', value=1, visible=True, step=1, minimum=0, maximum=100000)
        LR_POWER = gr.Slider(label='Lr Power', value=1.0, visible=True, step=0.1, minimum=0, maximum=1)
        LR_SCHEDULER = gr.Dropdown(label='Lr Scheduler', choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'], value='constant', visible=True)
        LR_WARMUP_STEPS = gr.Slider(label='Lr Warmup Steps', value=500, visible=True, step=1, minimum=0, maximum=100000)
        SCALE_LR = gr.Checkbox(label='Scale Lr', value=False, visible=True)
    with gr.Accordion(open=False, label="Logging") as LOGGING_ACCORDION:
        LOGGING_DIR = gr.Textbox(label='Logging Dir', value='logs', visible=False)
        REPORT_TO = gr.Dropdown(label='Report To', choices=['all', 'tensorboard', 'wandb'], value='tensorboard', visible=True)
        TRACKER_PROJECT_NAME = gr.Textbox(label='Tracker Project Name', value='sd_xl_train_t2iadapter', visible=True)
    with gr.Accordion(open=False, label="Model") as MODEL_ACCORDION:
        CONTROLNET_MODEL_NAME_OR_PATH = gr.Textbox(label='Controlnet Model Name Or Path', value='None', visible=True)
        PRETRAINED_VAE_MODEL_NAME_OR_PATH = gr.Textbox(label='Pretrained Vae Model Name Or Path', value='None', visible=True)
    with gr.Accordion(open=False, label="Optimizer") as OPTIMIZER_ACCORDION:
        ADAM_BETA1 = gr.Slider(label='Adam Beta1', value=0.9, visible=True, step=0.01, minimum=0, maximum=1)
        ADAM_BETA2 = gr.Slider(label='Adam Beta2', value=0.999, visible=True, step=0.01, minimum=0, maximum=1)
        ADAM_EPSILON = gr.Slider(label='Adam Epsilon', value=1e-08, visible=True, step=0.01, minimum=0, maximum=1)
        ADAM_WEIGHT_DECAY = gr.Slider(label='Adam Weight Decay', value=0.01, visible=True, step=0.01, minimum=0, maximum=1)
        MAX_GRAD_NORM = gr.Slider(label='Max Grad Norm', value=1.0, visible=True, step=0.1, minimum=0, maximum=1)
    with gr.Accordion(open=False, label="Performance") as PERFORMANCE_ACCORDION:
        ALLOW_TF32 = gr.Checkbox(label='Allow Tf32', value=True, visible=True)
        DATALOADER_NUM_WORKERS = gr.Slider(label='Dataloader Num Workers', value=1, visible=True, step=1, minimum=0, maximum=100)
        ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION = gr.Checkbox(label='Enable Xformers Memory Efficient Attention', value=True, visible=True)
        GRADIENT_CHECKPOINTING = gr.Checkbox(label='Gradient Checkpointing', value=True, visible=True)
        MIXED_PRECISION = gr.Dropdown(label='Mixed Precision', choices=['no', 'fp16', 'bf16'], value='bf16', visible=True)
        SEED = gr.Textbox(label='Seed', value='None', visible=True)
        SET_GRADS_TO_NONE = gr.Checkbox(label='Set Grads To None', value=False, visible=True)
        USE_8BIT_ADAM = gr.Checkbox(label='Use 8Bit Adam', value=True, visible=True)
    with gr.Accordion(open=False, label="Saving") as SAVING_ACCORDION:
        CHECKPOINTS_TOTAL_LIMIT = gr.Textbox(label='Checkpoints Total Limit', value='3', visible=True)
        HUB_MODEL_ID = gr.Textbox(label='Hub Model Id', value='None', visible=False)
        HUB_TOKEN = gr.Textbox(label='Hub Token', value='None', visible=False)
        PUSH_TO_HUB = gr.Checkbox(label='Push To Hub', value=False, visible=False)
    with gr.Accordion(open=False, label="Validation") as VALIDATION_ACCORDION:
        NUM_VALIDATION_IMAGES = gr.Slider(label='Num Validation Images', value=4, visible=True, step=1, minimum=0, maximum=100)
        VALIDATION_IMAGE = gr.Textbox(label='Validation Image', value='None', visible=True)
        VALIDATION_PROMPT = gr.Textbox(label='Validation Prompt', value='None', visible=True)
    manager.register_db_component("train_controlnet_sdxl", GRADIENT_ACCUMULATION_STEPS, "gradient_accumulation_steps", True, "Number of updates steps to accumulate before performing a backward/update pass.")
    manager.register_db_component("train_controlnet_sdxl", TRAIN_BATCH_SIZE, "train_batch_size", False, "Batch size (per device) for the training dataloader.")
    manager.register_db_component("train_controlnet_sdxl", CAPTION_COLUMN, "caption_column", False, "The column of the dataset containing a caption or a list of captions.")
    manager.register_db_component("train_controlnet_sdxl", CONDITIONING_IMAGE_COLUMN, "conditioning_image_column", False, "The column of the dataset containing the controlnet conditioning image.")
    manager.register_db_component("train_controlnet_sdxl", CROPS_COORDS_TOP_LEFT_H, "crops_coords_top_left_h", True, "Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet.")
    manager.register_db_component("train_controlnet_sdxl", CROPS_COORDS_TOP_LEFT_W, "crops_coords_top_left_w", True, "Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet.")
    manager.register_db_component("train_controlnet_sdxl", DATASET_CONFIG_NAME, "dataset_config_name", True, "The config of the Dataset, leave as None if there\'s only one config.")
    manager.register_db_component("train_controlnet_sdxl", DATASET_NAME, "dataset_name", False, "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private, dataset). It can also be a path pointing to a local copy of a dataset in your filesystem, or to a folder containing files that 🤗 Datasets can understand.")
    manager.register_db_component("train_controlnet_sdxl", IMAGE_COLUMN, "image_column", False, "The column of the dataset containing the target image.")
    manager.register_db_component("train_controlnet_sdxl", PROPORTION_EMPTY_PROMPTS, "proportion_empty_prompts", True, "Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).")
    manager.register_db_component("train_controlnet_sdxl", TRAIN_DATA_DIR, "train_data_dir", False, "A folder containing the training data. Folder contents must follow the structure described in https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file must exist to provide the captions for the images. Ignored if `dataset_name` is specified.")
    manager.register_db_component("train_controlnet_sdxl", RESOLUTION, "resolution", False, "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution")
    manager.register_db_component("train_controlnet_sdxl", CHECKPOINTING_STEPS, "checkpointing_steps", False, "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference.Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components.See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by stepinstructions.")
    manager.register_db_component("train_controlnet_sdxl", MAX_TRAIN_SAMPLES, "max_train_samples", True, "For debugging purposes or quicker training, truncate the number of training examples to this value if set.")
    manager.register_db_component("train_controlnet_sdxl", MAX_TRAIN_STEPS, "max_train_steps", False, "Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    manager.register_db_component("train_controlnet_sdxl", NUM_TRAIN_EPOCHS, "num_train_epochs", False, "")
    manager.register_db_component("train_controlnet_sdxl", VALIDATION_STEPS, "validation_steps", False, "Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images` and logging the images.")
    manager.register_db_component("train_controlnet_sdxl", LEARNING_RATE, "learning_rate", False, "Initial learning rate (after the potential warmup period) to use.")
    manager.register_db_component("train_controlnet_sdxl", LR_NUM_CYCLES, "lr_num_cycles", True, "Number of hard resets of the lr in cosine_with_restarts scheduler.")
    manager.register_db_component("train_controlnet_sdxl", LR_POWER, "lr_power", True, "Power factor of the polynomial scheduler.")
    manager.register_db_component("train_controlnet_sdxl", LR_SCHEDULER, "lr_scheduler", True, "The scheduler type to use. Choose between [\"linear\", \"cosine\", \"cosine_with_restarts\", \"polynomial\", \"constant\", \"constant_with_warmup\"]")
    manager.register_db_component("train_controlnet_sdxl", LR_WARMUP_STEPS, "lr_warmup_steps", True, "Number of steps for the warmup in the lr scheduler.")
    manager.register_db_component("train_controlnet_sdxl", SCALE_LR, "scale_lr", True, "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    manager.register_db_component("train_controlnet_sdxl", LOGGING_DIR, "logging_dir", False, "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    manager.register_db_component("train_controlnet_sdxl", REPORT_TO, "report_to", False, "The integration to report the results and logs to. Supported platforms are `\"tensorboard\"` (default), `\"wandb\"` and `\"comet_ml\"`. Use `\"all\"` to report to all integrations.")
    manager.register_db_component("train_controlnet_sdxl", TRACKER_PROJECT_NAME, "tracker_project_name", True, "The `project_name` argument passed to Accelerator.init_trackers for more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator")
    manager.register_db_component("train_controlnet_sdxl", CONTROLNET_MODEL_NAME_OR_PATH, "controlnet_model_name_or_path", False, "Path to pretrained controlnet model or model identifier from huggingface.co/models. If not specified controlnet weights are initialized from unet.")
    manager.register_db_component("train_controlnet_sdxl", PRETRAINED_VAE_MODEL_NAME_OR_PATH, "pretrained_vae_model_name_or_path", True, "Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.")
    manager.register_db_component("train_controlnet_sdxl", ADAM_BETA1, "adam_beta1", True, "The beta1 parameter for the Adam optimizer.")
    manager.register_db_component("train_controlnet_sdxl", ADAM_BETA2, "adam_beta2", True, "The beta2 parameter for the Adam optimizer.")
    manager.register_db_component("train_controlnet_sdxl", ADAM_EPSILON, "adam_epsilon", True, "Epsilon value for the Adam optimizer")
    manager.register_db_component("train_controlnet_sdxl", ADAM_WEIGHT_DECAY, "adam_weight_decay", True, "Weight decay to use.")
    manager.register_db_component("train_controlnet_sdxl", MAX_GRAD_NORM, "max_grad_norm", True, "Max gradient norm.")
    manager.register_db_component("train_controlnet_sdxl", ALLOW_TF32, "allow_tf32", True, "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    manager.register_db_component("train_controlnet_sdxl", DATALOADER_NUM_WORKERS, "dataloader_num_workers", True, "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    manager.register_db_component("train_controlnet_sdxl", ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION, "enable_xformers_memory_efficient_attention", False, "Whether or not to use xformers.")
    manager.register_db_component("train_controlnet_sdxl", GRADIENT_CHECKPOINTING, "gradient_checkpointing", True, "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    manager.register_db_component("train_controlnet_sdxl", MIXED_PRECISION, "mixed_precision", True, "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.")
    manager.register_db_component("train_controlnet_sdxl", SEED, "seed", True, "A seed for reproducible training.")
    manager.register_db_component("train_controlnet_sdxl", SET_GRADS_TO_NONE, "set_grads_to_none", True, "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain behaviors, so disable this argument if it causes any problems. More info: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html")
    manager.register_db_component("train_controlnet_sdxl", USE_8BIT_ADAM, "use_8bit_adam", True, "Whether or not to use 8-bit Adam from bitsandbytes.")
    manager.register_db_component("train_controlnet_sdxl", CHECKPOINTS_TOTAL_LIMIT, "checkpoints_total_limit", True, "Max number of checkpoints to store.")
    manager.register_db_component("train_controlnet_sdxl", HUB_MODEL_ID, "hub_model_id", False, "The name of the repository to keep in sync with the local `output_dir`.")
    manager.register_db_component("train_controlnet_sdxl", HUB_TOKEN, "hub_token", False, "The token to use to push to the Model Hub.")
    manager.register_db_component("train_controlnet_sdxl", PUSH_TO_HUB, "push_to_hub", False, "Whether or not to push the model to the Hub.")
    manager.register_db_component("train_controlnet_sdxl", NUM_VALIDATION_IMAGES, "num_validation_images", True, "Number of images to be generated for each `--validation_image`, `--validation_prompt` pair")
    manager.register_db_component("train_controlnet_sdxl", VALIDATION_IMAGE, "validation_image", False, "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps` and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a a single `--validation_prompt` to be used with all `--validation_image`s, or a single `--validation_image` that will be used with all `--validation_prompt`s.")
    manager.register_db_component("train_controlnet_sdxl", VALIDATION_PROMPT, "validation_prompt", True, "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`. Provide either a matching number of `--validation_image`s, a single `--validation_image` to be used with all prompts, or a single prompt that will be used with all `--validation_image`s.")