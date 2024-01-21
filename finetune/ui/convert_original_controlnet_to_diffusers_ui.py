import gradio as gr

from dreambooth.utils.ui_utils import ElementManager

manager = ElementManager()

CHECKPOINT_PATH = None
CROSS_ATTENTION_DIM = None
DEVICE = None
DUMP_PATH = None
EXTRACT_EMA = None
FROM_SAFETENSORS = None
IMAGE_SIZE = None
NUM_IN_CHANNELS = None
ORIGINAL_CONFIG_FILE = None
TO_SAFETENSORS = None
UPCAST_ATTENTION = None
USE_LINEAR_PROJECTION = None


def render():
    global CHECKPOINT_PATH
    global CROSS_ATTENTION_DIM
    global DEVICE
    global DUMP_PATH
    global EXTRACT_EMA
    global FROM_SAFETENSORS
    global IMAGE_SIZE
    global NUM_IN_CHANNELS
    global ORIGINAL_CONFIG_FILE
    global TO_SAFETENSORS
    global UPCAST_ATTENTION
    global USE_LINEAR_PROJECTION
    with gr.Accordion(open=False, label="Other") as OTHER_ACCORDION:
        CHECKPOINT_PATH = gr.Textbox(interactive=True, label='Checkpoint Path', value='None', visible=True)
        CROSS_ATTENTION_DIM = gr.Textbox(interactive=True, label='Cross Attention Dim', value='None', visible=True)
        DEVICE = gr.Textbox(interactive=True, label='Device', value='None', visible=True)
        DUMP_PATH = gr.Textbox(interactive=True, label='Dump Path', value='None', visible=True)
        EXTRACT_EMA = gr.Checkbox(interactive=True, label='Extract Ema', value=False, visible=True)
        FROM_SAFETENSORS = gr.Checkbox(interactive=True, label='From Safetensors', value=False, visible=True)
        IMAGE_SIZE = gr.Slider(interactive=True, label='Image Size', value=512, visible=True, step=1, minimum=0, maximum=100)
        NUM_IN_CHANNELS = gr.Textbox(interactive=True, label='Num In Channels', value='None', visible=True)
        ORIGINAL_CONFIG_FILE = gr.Textbox(interactive=True, label='Original Config File', value='None', visible=True)
        TO_SAFETENSORS = gr.Checkbox(interactive=True, label='To Safetensors', value=True, visible=True)
        UPCAST_ATTENTION = gr.Checkbox(interactive=True, label='Upcast Attention', value=False, visible=True)
        USE_LINEAR_PROJECTION = gr.Textbox(interactive=True, label='Use Linear Projection', value='None', visible=True)
    manager.register_db_component("convert_original_controlnet_to_diffusers", CHECKPOINT_PATH, "checkpoint_path", False, "Path to the checkpoint to convert.")
    manager.register_db_component("convert_original_controlnet_to_diffusers", CROSS_ATTENTION_DIM, "cross_attention_dim", False, "Override for cross attention_dim")
    manager.register_db_component("convert_original_controlnet_to_diffusers", DEVICE, "device", False, "Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    manager.register_db_component("convert_original_controlnet_to_diffusers", DUMP_PATH, "dump_path", False, "Path to the output model.")
    manager.register_db_component("convert_original_controlnet_to_diffusers", EXTRACT_EMA, "extract_ema", False, "Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning.")
    manager.register_db_component("convert_original_controlnet_to_diffusers", FROM_SAFETENSORS, "from_safetensors", False, "If `--checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.")
    manager.register_db_component("convert_original_controlnet_to_diffusers", IMAGE_SIZE, "image_size", False, "The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Siffusion v2 Base. Use 768 for Stable Diffusion v2.")
    manager.register_db_component("convert_original_controlnet_to_diffusers", NUM_IN_CHANNELS, "num_in_channels", False, "The number of input channels. If `None` number of input channels will be automatically inferred.")
    manager.register_db_component("convert_original_controlnet_to_diffusers", ORIGINAL_CONFIG_FILE, "original_config_file", False, "The YAML config file corresponding to the original architecture.")
    manager.register_db_component("convert_original_controlnet_to_diffusers", TO_SAFETENSORS, "to_safetensors", False, "Whether to store pipeline in safetensors format or not.")
    manager.register_db_component("convert_original_controlnet_to_diffusers", UPCAST_ATTENTION, "upcast_attention", False, "Whether the attention computation should always be upcasted. This is necessary when running stable diffusion 2.1.")
    manager.register_db_component("convert_original_controlnet_to_diffusers", USE_LINEAR_PROJECTION, "use_linear_projection", False, "Override for use linear projection")