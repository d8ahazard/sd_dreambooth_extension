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
VALIDATION_EPOCHS = None
RANK = None
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
PRE_COMPUTE_TEXT_EMBEDDINGS = None
PRIOR_GENERATION_PRECISION = None
SEED = None
TEXT_ENCODER_USE_ATTENTION_MASK = None
TOKENIZER_MAX_LENGTH = None
TRAIN_TEXT_ENCODER = None
USE_8BIT_ADAM = None
CHECKPOINTS_TOTAL_LIMIT = None
HUB_MODEL_ID = None
HUB_TOKEN = None
PUSH_TO_HUB = None
NUM_VALIDATION_IMAGES = None
VALIDATION_IMAGES = None
VALIDATION_PROMPT = None


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
    global VALIDATION_EPOCHS
    global RANK
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
    global PRE_COMPUTE_TEXT_EMBEDDINGS
    global PRIOR_GENERATION_PRECISION
    global SEED
    global TEXT_ENCODER_USE_ATTENTION_MASK
    global TOKENIZER_MAX_LENGTH
    global TRAIN_TEXT_ENCODER
    global USE_8BIT_ADAM
    global CHECKPOINTS_TOTAL_LIMIT
    global HUB_MODEL_ID
    global HUB_TOKEN
    global PUSH_TO_HUB
    global NUM_VALIDATION_IMAGES
    global VALIDATION_IMAGES
    global VALIDATION_PROMPT
    with gr.Accordion(open=False, label="Batching") as BATCHING_ACCORDION:
        GRADIENT_ACCUMULATION_STEPS = gr.Slider(label='Gradient Accumulation Steps', value=1, visible=True, step=1, minimum=0, maximum=100)
        SAMPLE_BATCH_SIZE = gr.Slider(label='Sample Batch Size', value=4, visible=True, step=1, minimum=1, maximum=1000)
        TRAIN_BATCH_SIZE = gr.Slider(label='Train Batch Size', value=4, visible=True, step=1, minimum=1, maximum=1000)
    with gr.Accordion(open=False, label="Dataset") as DATASET_ACCORDION:
        CLASS_DATA_DIR = gr.Textbox(label='Class Data Dir', value='None', visible=True)
        CLASS_LABELS_CONDITIONING = gr.Textbox(label='Class Labels Conditioning', value='None', visible=True)
        CLASS_PROMPT = gr.Textbox(label='Class Prompt', value='None', visible=True)
        INSTANCE_DATA_DIR = gr.Textbox(label='Instance Data Dir', value='None', visible=True)
        INSTANCE_PROMPT = gr.Textbox(label='Instance Prompt', value='None', visible=True)
        NUM_CLASS_IMAGES = gr.Slider(label='Num Class Images', value=100, visible=True, step=1, minimum=0, maximum=100000)
        PRIOR_LOSS_WEIGHT = gr.Slider(label='Prior Loss Weight', value=1.0, visible=True, step=0.1, minimum=0, maximum=1)
        WITH_PRIOR_PRESERVATION = gr.Checkbox(label='With Prior Preservation', value=False, visible=True)
    with gr.Accordion(open=False, label="Image Processing") as IMAGE_PROCESSING_ACCORDION:
        CENTER_CROP = gr.Checkbox(label='Center Crop', value=False, visible=True)
        RESOLUTION = gr.Slider(label='Resolution', value=1024, visible=True, step=64, minimum=256, maximum=4096)
    with gr.Accordion(open=False, label="Intervals") as INTERVALS_ACCORDION:
        CHECKPOINTING_STEPS = gr.Slider(label='Checkpointing Steps', value=500, visible=True, step=1, minimum=0, maximum=100000)
        MAX_TRAIN_STEPS = gr.Textbox(label='Max Train Steps', value='None', visible=False)
        NUM_TRAIN_EPOCHS = gr.Slider(label='Num Train Epochs', value=1, visible=True, step=1, minimum=0, maximum=100000)
        VALIDATION_EPOCHS = gr.Slider(label='Validation Epochs', value=1, visible=True, step=1, minimum=0, maximum=100)
    with gr.Accordion(open=False, label="Lora") as LORA_ACCORDION:
        RANK = gr.Slider(label='Rank', value=4, visible=True, step=1, minimum=0, maximum=100)
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
        LOCAL_RANK = gr.Slider(label='Local Rank', value=-1, visible=True, step=1, minimum=0, maximum=100)
        MIXED_PRECISION = gr.Dropdown(label='Mixed Precision', choices=['no', 'fp16', 'bf16'], value='bf16', visible=True)
        PRE_COMPUTE_TEXT_EMBEDDINGS = gr.Checkbox(label='Pre Compute Text Embeddings', value=False, visible=True)
        PRIOR_GENERATION_PRECISION = gr.Dropdown(label='Prior Generation Precision', choices=['no', 'fp32', 'fp16', 'bf16'], value='bf16', visible=True)
        SEED = gr.Textbox(label='Seed', value='None', visible=True)
        TEXT_ENCODER_USE_ATTENTION_MASK = gr.Checkbox(label='Text Encoder Use Attention Mask', value=False, visible=True)
        TOKENIZER_MAX_LENGTH = gr.Textbox(label='Tokenizer Max Length', value='None', visible=True)
        TRAIN_TEXT_ENCODER = gr.Checkbox(label='Train Text Encoder', value=False, visible=True)
        USE_8BIT_ADAM = gr.Checkbox(label='Use 8Bit Adam', value=True, visible=True)
    with gr.Accordion(open=False, label="Saving") as SAVING_ACCORDION:
        CHECKPOINTS_TOTAL_LIMIT = gr.Textbox(label='Checkpoints Total Limit', value='3', visible=True)
        HUB_MODEL_ID = gr.Textbox(label='Hub Model Id', value='None', visible=False)
        HUB_TOKEN = gr.Textbox(label='Hub Token', value='None', visible=False)
        PUSH_TO_HUB = gr.Checkbox(label='Push To Hub', value=False, visible=False)
    with gr.Accordion(open=False, label="Validation") as VALIDATION_ACCORDION:
        NUM_VALIDATION_IMAGES = gr.Slider(label='Num Validation Images', value=4, visible=True, step=1, minimum=0, maximum=100)
        VALIDATION_IMAGES = gr.Textbox(label='Validation Images', value='None', visible=True)
        VALIDATION_PROMPT = gr.Textbox(label='Validation Prompt', value='None', visible=True)
    manager.register_db_component("train_dreambooth_lora", GRADIENT_ACCUMULATION_STEPS, "gradient_accumulation_steps", True, "Number of updates steps to accumulate before performing a backward/update pass.")
    manager.register_db_component("train_dreambooth_lora", SAMPLE_BATCH_SIZE, "sample_batch_size", True, "Batch size (per device) for sampling images.")
    manager.register_db_component("train_dreambooth_lora", TRAIN_BATCH_SIZE, "train_batch_size", False, "Batch size (per device) for the training dataloader.")
    manager.register_db_component("train_dreambooth_lora", CLASS_DATA_DIR, "class_data_dir", False, "A folder containing the training data of class images.")
    manager.register_db_component("train_dreambooth_lora", CLASS_LABELS_CONDITIONING, "class_labels_conditioning", False, "The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.")
    manager.register_db_component("train_dreambooth_lora", CLASS_PROMPT, "class_prompt", False, "The prompt to specify images in the same class as provided instance images.")
    manager.register_db_component("train_dreambooth_lora", INSTANCE_DATA_DIR, "instance_data_dir", False, "A folder containing the training data of instance images.")
    manager.register_db_component("train_dreambooth_lora", INSTANCE_PROMPT, "instance_prompt", False, "The prompt with identifier specifying the instance")
    manager.register_db_component("train_dreambooth_lora", NUM_CLASS_IMAGES, "num_class_images", False, "Minimal class images for prior preservation loss. If there are not enough images already present in class_data_dir, additional images will be sampled with class_prompt.")
    manager.register_db_component("train_dreambooth_lora", PRIOR_LOSS_WEIGHT, "prior_loss_weight", True, "The weight of prior preservation loss.")
    manager.register_db_component("train_dreambooth_lora", WITH_PRIOR_PRESERVATION, "with_prior_preservation", True, "Flag to add prior preservation loss.")
    manager.register_db_component("train_dreambooth_lora", CENTER_CROP, "center_crop", True, "Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.")
    manager.register_db_component("train_dreambooth_lora", RESOLUTION, "resolution", False, "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution")
    manager.register_db_component("train_dreambooth_lora", CHECKPOINTING_STEPS, "checkpointing_steps", False, "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final checkpoints in case they are better than the last checkpoint, and are also suitable for resuming training using `--resume_from_checkpoint`.")
    manager.register_db_component("train_dreambooth_lora", MAX_TRAIN_STEPS, "max_train_steps", False, "Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    manager.register_db_component("train_dreambooth_lora", NUM_TRAIN_EPOCHS, "num_train_epochs", False, "")
    manager.register_db_component("train_dreambooth_lora", VALIDATION_EPOCHS, "validation_epochs", False, "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images`.")
    manager.register_db_component("train_dreambooth_lora", RANK, "rank", True, "The dimension of the LoRA update matrices.")
    manager.register_db_component("train_dreambooth_lora", LEARNING_RATE, "learning_rate", False, "Initial learning rate (after the potential warmup period) to use.")
    manager.register_db_component("train_dreambooth_lora", LR_NUM_CYCLES, "lr_num_cycles", True, "Number of hard resets of the lr in cosine_with_restarts scheduler.")
    manager.register_db_component("train_dreambooth_lora", LR_POWER, "lr_power", True, "Power factor of the polynomial scheduler.")
    manager.register_db_component("train_dreambooth_lora", LR_SCHEDULER, "lr_scheduler", True, "The scheduler type to use. Choose between [\"linear\", \"cosine\", \"cosine_with_restarts\", \"polynomial\", \"constant\", \"constant_with_warmup\"]")
    manager.register_db_component("train_dreambooth_lora", LR_WARMUP_STEPS, "lr_warmup_steps", True, "Number of steps for the warmup in the lr scheduler.")
    manager.register_db_component("train_dreambooth_lora", SCALE_LR, "scale_lr", True, "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    manager.register_db_component("train_dreambooth_lora", LOGGING_DIR, "logging_dir", False, "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    manager.register_db_component("train_dreambooth_lora", REPORT_TO, "report_to", False, "The integration to report the results and logs to. Supported platforms are `\"tensorboard\"` (default), `\"wandb\"` and `\"comet_ml\"`. Use `\"all\"` to report to all integrations.")
    manager.register_db_component("train_dreambooth_lora", ADAM_BETA1, "adam_beta1", True, "The beta1 parameter for the Adam optimizer.")
    manager.register_db_component("train_dreambooth_lora", ADAM_BETA2, "adam_beta2", True, "The beta2 parameter for the Adam optimizer.")
    manager.register_db_component("train_dreambooth_lora", ADAM_EPSILON, "adam_epsilon", True, "Epsilon value for the Adam optimizer")
    manager.register_db_component("train_dreambooth_lora", ADAM_WEIGHT_DECAY, "adam_weight_decay", True, "Weight decay to use.")
    manager.register_db_component("train_dreambooth_lora", MAX_GRAD_NORM, "max_grad_norm", True, "Max gradient norm.")
    manager.register_db_component("train_dreambooth_lora", ALLOW_TF32, "allow_tf32", True, "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    manager.register_db_component("train_dreambooth_lora", DATALOADER_NUM_WORKERS, "dataloader_num_workers", True, "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    manager.register_db_component("train_dreambooth_lora", ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION, "enable_xformers_memory_efficient_attention", False, "Whether or not to use xformers.")
    manager.register_db_component("train_dreambooth_lora", GRADIENT_CHECKPOINTING, "gradient_checkpointing", True, "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    manager.register_db_component("train_dreambooth_lora", LOCAL_RANK, "local_rank", True, "For distributed training: local_rank")
    manager.register_db_component("train_dreambooth_lora", MIXED_PRECISION, "mixed_precision", True, "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.")
    manager.register_db_component("train_dreambooth_lora", PRE_COMPUTE_TEXT_EMBEDDINGS, "pre_compute_text_embeddings", True, "Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.")
    manager.register_db_component("train_dreambooth_lora", PRIOR_GENERATION_PRECISION, "prior_generation_precision", True, "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32.")
    manager.register_db_component("train_dreambooth_lora", SEED, "seed", True, "A seed for reproducible training.")
    manager.register_db_component("train_dreambooth_lora", TEXT_ENCODER_USE_ATTENTION_MASK, "text_encoder_use_attention_mask", True, "Whether to use attention mask for the text encoder")
    manager.register_db_component("train_dreambooth_lora", TOKENIZER_MAX_LENGTH, "tokenizer_max_length", True, "The maximum length of the tokenizer. If not set, will default to the tokenizer\'s max length.")
    manager.register_db_component("train_dreambooth_lora", TRAIN_TEXT_ENCODER, "train_text_encoder", True, "Whether to train the text encoder. If set, the text encoder should be float32 precision.")
    manager.register_db_component("train_dreambooth_lora", USE_8BIT_ADAM, "use_8bit_adam", True, "Whether or not to use 8-bit Adam from bitsandbytes.")
    manager.register_db_component("train_dreambooth_lora", CHECKPOINTS_TOTAL_LIMIT, "checkpoints_total_limit", True, "Max number of checkpoints to store.")
    manager.register_db_component("train_dreambooth_lora", HUB_MODEL_ID, "hub_model_id", False, "The name of the repository to keep in sync with the local `output_dir`.")
    manager.register_db_component("train_dreambooth_lora", HUB_TOKEN, "hub_token", False, "The token to use to push to the Model Hub.")
    manager.register_db_component("train_dreambooth_lora", PUSH_TO_HUB, "push_to_hub", False, "Whether or not to push the model to the Hub.")
    manager.register_db_component("train_dreambooth_lora", NUM_VALIDATION_IMAGES, "num_validation_images", True, "Number of images that should be generated during validation with `validation_prompt`.")
    manager.register_db_component("train_dreambooth_lora", VALIDATION_IMAGES, "validation_images", True, "Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.")
    manager.register_db_component("train_dreambooth_lora", VALIDATION_PROMPT, "validation_prompt", True, "A prompt that is used during validation to verify that the model is learning.")