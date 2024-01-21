import gradio as gr

from dreambooth.utils.ui_utils import ElementManager

manager = ElementManager()

CHECKPOINT_PATH = None
CLIP_STATS_PATH = None
CONFIG_FILES = None
CONTROLNET = None
DEVICE = None
DUMP_PATH = None
EXTRACT_EMA = None
FROM_SAFETENSORS = None
HALF = None
IMAGE_SIZE = None
NUM_IN_CHANNELS = None
ORIGINAL_CONFIG_FILE = None
PIPELINE_CLASS_NAME = None
PIPELINE_TYPE = None
SCHEDULER_TYPE = None
STABLE_UNCLIP = None
STABLE_UNCLIP_PRIOR = None
TO_SAFETENSORS = None
UPCAST_ATTENTION = None
VAE_PATH = None
PREDICTION_TYPE = None


def render():
    global CHECKPOINT_PATH
    global CLIP_STATS_PATH
    global CONFIG_FILES
    global CONTROLNET
    global DEVICE
    global DUMP_PATH
    global EXTRACT_EMA
    global FROM_SAFETENSORS
    global HALF
    global IMAGE_SIZE
    global NUM_IN_CHANNELS
    global ORIGINAL_CONFIG_FILE
    global PIPELINE_CLASS_NAME
    global PIPELINE_TYPE
    global SCHEDULER_TYPE
    global STABLE_UNCLIP
    global STABLE_UNCLIP_PRIOR
    global TO_SAFETENSORS
    global UPCAST_ATTENTION
    global VAE_PATH
    global PREDICTION_TYPE
    with gr.Accordion(open=False, label="Other") as OTHER_ACCORDION:
        CHECKPOINT_PATH = gr.Textbox(interactive=True, label='Checkpoint Path', value='None', visible=True)
        CLIP_STATS_PATH = gr.Textbox(interactive=True, label='Clip Stats Path', value='None', visible=True)
        CONFIG_FILES = gr.Textbox(interactive=True, label='Config Files', value='None', visible=True)
        CONTROLNET = gr.Textbox(interactive=True, label='Controlnet', value='None', visible=True)
        DEVICE = gr.Textbox(interactive=True, label='Device', value='None', visible=True)
        DUMP_PATH = gr.Textbox(interactive=True, label='Dump Path', value='None', visible=True)
        EXTRACT_EMA = gr.Checkbox(interactive=True, label='Extract Ema', value=False, visible=True)
        FROM_SAFETENSORS = gr.Checkbox(interactive=True, label='From Safetensors', value=False, visible=True)
        HALF = gr.Checkbox(interactive=True, label='Half', value=False, visible=True)
        IMAGE_SIZE = gr.Slider(interactive=True, label='Image Size', value=512, visible=True, step=1, minimum=0, maximum=100)
        NUM_IN_CHANNELS = gr.Textbox(interactive=True, label='Num In Channels', value='None', visible=True)
        ORIGINAL_CONFIG_FILE = gr.Textbox(interactive=True, label='Original Config File', value='None', visible=True)
        PIPELINE_CLASS_NAME = gr.Textbox(interactive=True, label='Pipeline Class Name', value='None', visible=True)
        PIPELINE_TYPE = gr.Textbox(interactive=True, label='Pipeline Type', value='None', visible=True)
        SCHEDULER_TYPE = gr.Textbox(interactive=True, label='Scheduler Type', value='pndm', visible=True)
        STABLE_UNCLIP = gr.Textbox(interactive=True, label='Stable Unclip', value='None', visible=True)
        STABLE_UNCLIP_PRIOR = gr.Textbox(interactive=True, label='Stable Unclip Prior', value='None', visible=True)
        TO_SAFETENSORS = gr.Checkbox(interactive=True, label='To Safetensors', value=True, visible=True)
        UPCAST_ATTENTION = gr.Checkbox(interactive=True, label='Upcast Attention', value=False, visible=True)
        VAE_PATH = gr.Textbox(interactive=True, label='Vae Path', value='None', visible=True)
    with gr.Accordion(open=False, label="Performance") as PERFORMANCE_ACCORDION:
        PREDICTION_TYPE = gr.Dropdown(interactive=True, label='Prediction Type', choices=['epsilon', 'v_prediction'], value='None', visible=True)
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", CHECKPOINT_PATH, "checkpoint_path", False, "Path to the checkpoint to convert.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", CLIP_STATS_PATH, "clip_stats_path", False, "Path to the clip stats file. Only required if the stable unclip model\'s config specifies `model.params.noise_aug_config.params.clip_stats_path`.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", CONFIG_FILES, "config_files", False, "The YAML config file corresponding to the architecture.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", CONTROLNET, "controlnet", False, "Set flag if this is a controlnet checkpoint.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", DEVICE, "device", False, "Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", DUMP_PATH, "dump_path", False, "Path to the output model.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", EXTRACT_EMA, "extract_ema", False, "Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", FROM_SAFETENSORS, "from_safetensors", False, "If `--checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", HALF, "half", False, "Save weights in half precision.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", IMAGE_SIZE, "image_size", False, "The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Siffusion v2 Base. Use 768 for Stable Diffusion v2.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", NUM_IN_CHANNELS, "num_in_channels", False, "The number of input channels. If `None` number of input channels will be automatically inferred.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", ORIGINAL_CONFIG_FILE, "original_config_file", False, "The YAML config file corresponding to the original architecture.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", PIPELINE_CLASS_NAME, "pipeline_class_name", False, "Specify the pipeline class name")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", PIPELINE_TYPE, "pipeline_type", False, "The pipeline type. One of \'FrozenOpenCLIPEmbedder\', \'FrozenCLIPEmbedder\', \'PaintByExample\'. If `None` pipeline will be automatically inferred.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", SCHEDULER_TYPE, "scheduler_type", False, "Type of scheduler to use. Should be one of [\'pndm\', \'lms\', \'ddim\', \'euler\', \'euler-ancestral\', \'dpm\']")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", STABLE_UNCLIP, "stable_unclip", False, "Set if this is a stable unCLIP model. One of \'txt2img\' or \'img2img\'.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", STABLE_UNCLIP_PRIOR, "stable_unclip_prior", False, "Set if this is a stable unCLIP txt2img model. Selects which prior to use. If `--stable_unclip` is set to `txt2img`, the karlo prior (https://huggingface.co/kakaobrain/karlo-v1-alpha/tree/main/prior) is selected by default.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", TO_SAFETENSORS, "to_safetensors", False, "Whether to store pipeline in safetensors format or not.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", UPCAST_ATTENTION, "upcast_attention", False, "Whether the attention computation should always be upcasted. This is necessary when running stable diffusion 2.1.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", VAE_PATH, "vae_path", False, "Set to a path, hub id to an already converted vae to not convert it again.")
    manager.register_db_component("convert_original_stable_diffusion_to_diffusers", PREDICTION_TYPE, "prediction_type", True, "The prediction type that the model was trained on. Use \'epsilon\' for Stable Diffusion v1.X and Stable Diffusion v2 Base. Use \'v_prediction\' for Stable Diffusion v2.")