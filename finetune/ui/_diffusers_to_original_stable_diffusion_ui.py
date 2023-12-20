import gradio as gr

from dreambooth.utils.ui_utils import ElementManager

manager = ElementManager()

CHECKPOINT_PATH = None
HALF = None
MODEL_PATH = None
USE_SAFETENSORS = None


def render():
    global CHECKPOINT_PATH
    global HALF
    global MODEL_PATH
    global USE_SAFETENSORS
    with gr.Accordion(open=False, label="Other") as OTHER_ACCORDION:
        CHECKPOINT_PATH = gr.Textbox(label='Checkpoint Path', value='None', visible=True)
        HALF = gr.Checkbox(label='Half', value=False, visible=True)
        MODEL_PATH = gr.Textbox(label='Model Path', value='None', visible=True)
        USE_SAFETENSORS = gr.Checkbox(label='Use Safetensors', value=False, visible=True)
    manager.register_db_component("_diffusers_to_original_stable_diffusion", CHECKPOINT_PATH, "checkpoint_path", False, "Path to the output model.")
    manager.register_db_component("_diffusers_to_original_stable_diffusion", HALF, "half", False, "Save weights in half precision.")
    manager.register_db_component("_diffusers_to_original_stable_diffusion", MODEL_PATH, "model_path", False, "Path to the model to convert.")
    manager.register_db_component("_diffusers_to_original_stable_diffusion", USE_SAFETENSORS, "use_safetensors", False, "Save weights use safetensors, default is ckpt.")