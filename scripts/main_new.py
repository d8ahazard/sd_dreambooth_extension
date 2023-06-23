import importlib
import json
import time
from typing import List

import gradio as gr
from gradio.inputs import Slider, Dropdown, Number, Checkbox, Radio, Textbox
from gradio.outputs import Text
from dreambooth.dataclasses.db_config import from_file, save_config
from dreambooth.diff_to_sd import compile_checkpoint
from dreambooth.secret import (
    get_secret,
    create_secret,
    clear_secret,
)
from dreambooth.shared import (
    status,
    get_launch_errors,
)
from dreambooth.ui_functions import (
    performance_wizard,
    training_wizard,
    training_wizard_person,
    load_model_params,
    ui_classifiers,
    debug_buckets,
    create_model,
    generate_samples,
    load_params,
    start_training,
    update_extension,
    start_crop,
)
from dreambooth.utils.image_utils import (
    get_scheduler_names,
)
from dreambooth.utils.model_utils import (
    get_db_models,
    get_sorted_lora_models,
    get_model_snapshots,
    get_shared_models,
)
from dreambooth.utils.utils import (
    list_attention,
    list_precisions,
    wrap_gpu_call,
    printm,
    list_optimizer,
    list_schedulers,
)
from dreambooth.webhook import save_and_test_webhook
from helpers.log_parser import LogParser
from helpers.version_helper import check_updates
from modules import script_callbacks, sd_models
from modules.ui import gr_show, create_refresh_button

params_to_save = []
params_to_load = []
refresh_symbol = "\U0001f504"  # 🔄
delete_symbol = "\U0001F5D1"  # 🗑️
update_symbol = "\U0001F51D"  # 🠝
log_parser = LogParser()


def get_sd_models():
    sd_models.list_models()
    sd_list = sd_models.checkpoints_list
    names = []
    for key in sd_list:
        names.append(key)
    return names


def calc_time_left(progress, threshold, label, force_display):
    if progress == 0:
        return ""
    else:
        if status.time_start is None:
            time_since_start = 0
        else:
            time_since_start = time.time() - status.time_start
        eta = time_since_start / progress
        eta_relative = eta - time_since_start
        if (eta_relative > threshold and progress > 0.02) or force_display:
            if eta_relative > 86400:
                days = eta_relative // 86400
                remainder = days * 86400
                eta_relative -= remainder
                return f"{label}{days}:{time.strftime('%H:%M:%S', time.gmtime(eta_relative))}"
            if eta_relative > 3600:
                return label + time.strftime("%H:%M:%S", time.gmtime(eta_relative))
            elif eta_relative > 60:
                return label + time.strftime("%M:%S", time.gmtime(eta_relative))
            else:
                return label + time.strftime("%Ss", time.gmtime(eta_relative))
        else:
            return ""


def has_face_swap():
    script_class = None
    try:
        from modules.scripts import list_scripts

        scripts = list_scripts("scripts", ".py")
        for script_file in scripts:
            if script_file.filename == "batch_face_swap.py":
                path = script_file.path
                module_name = "batch_face_swap"
                spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                script_class = module.Script
                break
    except Exception as f:
        print(f"Can't check face swap: {f}")
    return script_class is not None


def check_progress_call():
    """
    Check the progress from share dreamstate and return appropriate UI elements.
    @return:
    active: Checkbox to physically hold an active state
    pspan: Progress bar span contents
    preview: Preview Image/Visibility
    gallery: Gallery Image/Visibility
    textinfo_result: Primary status
    sample_prompts: List = A list of prompts corresponding with gallery contents
    check_progress_initial: Hides the manual 'check progress' button
    """
    active_box = gr.update(value=status.active)
    if not status.active:
        return (
            active_box,
            "",
            gr.update(visible=False, value=None),
            gr.update(visible=True),
            gr_show(True),
            gr_show(True),
            gr_show(False),
        )

    progress = 0

    if status.job_count > 0:
        progress += status.job_no / status.job_count

    time_left = calc_time_left(progress, 1, " ETA: ", status.time_left_force_display)
    if time_left:
        status.time_left_force_display = True

    progress = min(progress, 1)
    progressbar = f"""<div class='progressDiv'><div class='progress' style="overflow:visible;width:{progress * 100}%;white-space:nowrap;">{"&nbsp;" * 2 + str(int(progress * 100)) + "%" + time_left if progress > 0.01 else ""}</div></div>"""
    status.set_current_image()
    image = status.current_image
    preview = None
    gallery = None

    if image is None:
        preview = gr.update(visible=False, value=None)
        gallery = gr.update(visible=True)
    else:
        if isinstance(image, List):
            if len(image) > 1:
                status.current_image = None
                preview = gr.update(visible=False, value=None)
                gallery = gr.update(visible=True, value=image)
            elif len(image) == 1:
                preview = gr.update(visible=True, value=image[0])
                gallery = gr.update(visible=True, value=None)
        else:
            preview = gr.update(visible=True, value=image)
            gallery = gr.update(visible=True, value=None)

    if status.textinfo is not None:
        textinfo_result = status.textinfo
    else:
        textinfo_result = ""

    if status.textinfo2 is not None:
        textinfo_result = f"{textinfo_result}<br>{status.textinfo2}"

    prompts = ""
    if len(status.sample_prompts) > 0:
        if len(status.sample_prompts) > 1:
            prompts = "<br>".join(status.sample_prompts)
        else:
            prompts = status.sample_prompts[0]

    pspan = f"<span id='db_progress_span' style='display: none'>{time.time()}</span><p>{progressbar}</p>"
    return (
        active_box,
        pspan,
        preview,
        gallery,
        textinfo_result,
        gr.update(value=prompts),
        gr_show(False),
    )


def check_progress_call_initial():
    status.begin()
    (
        active_box,
        pspan,
        preview,
        gallery,
        textinfo_result,
        prompts_result,
        pbutton_result,
    ) = check_progress_call()
    return (
        active_box,
        pspan,
        gr_show(False),
        gr.update(value=[]),
        textinfo_result,
        gr.update(value=[]),
        gr_show(False),
    )


def ui_gen_ckpt(model_name: str):
    if isinstance(model_name, List):
        model_name = model_name[0]
    if model_name == "" or model_name is None:
        return "Please select a model."
    config = from_file(model_name)
    printm("Config loaded")
    lora_path = config.lora_model_name
    print(f"Lora path: {lora_path}")
    res = compile_checkpoint(model_name, lora_path, True, True, config.checkpoint)
    return res

def create_element(data):
    if data['type'] == 'ConstrainedIntValue':
        return Slider(minimum=data['min'], maximum=data['max'], step=1, default=data['value'], label=data['title'])
    elif data['type'] == 'ConstrainedFloatValue':
        return Slider(minimum=data['min'], maximum=data['max'], step=data['step'], default=data['value'], label=data['title'])
    elif data['type'] == 'str' and 'options' in data:
        return Dropdown(choices=data['options'], default=data['value'], label=data['title'])
    elif data['type'] == 'str':
        return Textbox(default=data['value'], label=data['title'])
    elif data['type'] == 'bool':
        return Checkbox(label=data['title'], default=data['value'])
    elif data['type'].endswith('_modelSelect'):
        # You will need to replace this with your custom UI element
        pass

def create_gradio_ui(json_str):
    data = json.loads(json_str)
    groups = {}
    general_container = []

    for k, v in data.items():
        if k not in ["keys", "schedulers", "optimizers", "precisions", "attentions"] and not v['description'].startswith('[model]'):
            element = create_element(v)
            if element:
                classes = []
                if v['description'].startswith('['):
                    end_bracket_index = v['description'].index(']')
                    classes = v['description'][1:end_bracket_index].split(',')
                    classes = [class_ + 'Only' for class_ in classes]
                    element.classes = classes

                if 'group' in v:
                    group_name = v['group']
                    if group_name not in groups:
                        groups[group_name] = gr.outputs.Accordion(sections={}, open=False)
                    groups[group_name].add_section(v['title'], element)
                else:
                    general_container.append(element)

    for group_name, group in groups.items():
        general_container.append(group)

    return gr.Interface(fn=lambda *args: "This is a placeholder function", inputs=general_container, outputs=Text())

def on_ui_tabs():
    interface = create_gradio_ui(json_str)

    with gr.Blocks() as dreambooth_interface:
        # Top button row
        with gr.Row(equal_height=True, elem_id="DbTopRow"):
            db_load_params = gr.Button(value="Load Settings", elem_id="db_load_params")
            db_save_params = gr.Button(value="Save Settings", elem_id="db_save_config")
            db_train_model = gr.Button(
                value="Train", variant="primary", elem_id="db_train"
            )
            db_generate_checkpoint = gr.Button(
                value="Generate Ckpt", elem_id="db_gen_ckpt"
            )
            db_generate_checkpoint_during = gr.Button(
                value="Save Weights", elem_id="db_gen_ckpt_during"
            )
            db_train_sample = gr.Button(
                value="Generate Samples", elem_id="db_train_sample"
            )
            db_cancel = gr.Button(value="Cancel", elem_id="db_cancel")
        with gr.Row():
            gr.HTML(value="Select or create a model to begin.", elem_id="hint_row")
        with gr.Row().style(equal_height=False):
            with gr.Column(variant="panel", elem_id="MainPanel") as main_panel:
                for input_interface in interface.input_interfaces:
                    main_panel.append(input_interface[1])

    return ((dreambooth_interface, "Dreambooth", "dreambooth_interface"),)
