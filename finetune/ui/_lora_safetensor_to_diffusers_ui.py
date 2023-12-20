import gradio as gr

from dreambooth.utils.ui_utils import ElementManager

manager = ElementManager()

ALPHA = None
BASE_MODEL_PATH = None
CHECKPOINT_PATH = None
DEVICE = None
DUMP_PATH = None
LORA_PREFIX_TEXT_ENCODER = None
LORA_PREFIX_UNET = None
TO_SAFETENSORS = None


def render():
    global ALPHA
    global BASE_MODEL_PATH
    global CHECKPOINT_PATH
    global DEVICE
    global DUMP_PATH
    global LORA_PREFIX_TEXT_ENCODER
    global LORA_PREFIX_UNET
    global TO_SAFETENSORS
    with gr.Accordion(open=False, label="Other") as OTHER_ACCORDION:
        ALPHA = gr.Slider(label='Alpha', value=0.75, visible=True, step=0.01, minimum=0, maximum=1)
        BASE_MODEL_PATH = gr.Textbox(label='Base Model Path', value='None', visible=True)
        CHECKPOINT_PATH = gr.Textbox(label='Checkpoint Path', value='None', visible=True)
        DEVICE = gr.Textbox(label='Device', value='None', visible=True)
        DUMP_PATH = gr.Textbox(label='Dump Path', value='None', visible=True)
        LORA_PREFIX_TEXT_ENCODER = gr.Textbox(label='Lora Prefix Text Encoder', value='lora_te', visible=True)
        LORA_PREFIX_UNET = gr.Textbox(label='Lora Prefix Unet', value='lora_unet', visible=True)
        TO_SAFETENSORS = gr.Checkbox(label='To Safetensors', value=False, visible=True)
    manager.register_db_component("_lora_safetensor_to_diffusers", ALPHA, "alpha", False, "The merging ratio in W = W0 + alpha * deltaW")
    manager.register_db_component("_lora_safetensor_to_diffusers", BASE_MODEL_PATH, "base_model_path", False, "Path to the base model in diffusers format.")
    manager.register_db_component("_lora_safetensor_to_diffusers", CHECKPOINT_PATH, "checkpoint_path", False, "Path to the checkpoint to convert.")
    manager.register_db_component("_lora_safetensor_to_diffusers", DEVICE, "device", False, "Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    manager.register_db_component("_lora_safetensor_to_diffusers", DUMP_PATH, "dump_path", False, "Path to the output model.")
    manager.register_db_component("_lora_safetensor_to_diffusers", LORA_PREFIX_TEXT_ENCODER, "lora_prefix_text_encoder", False, "The prefix of text encoder weight in safetensors")
    manager.register_db_component("_lora_safetensor_to_diffusers", LORA_PREFIX_UNET, "lora_prefix_unet", False, "The prefix of UNet weight in safetensors")
    manager.register_db_component("_lora_safetensor_to_diffusers", TO_SAFETENSORS, "to_safetensors", False, "Whether to store pipeline in safetensors format or not.")