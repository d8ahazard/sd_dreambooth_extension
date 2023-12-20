import importlib
import json
import os

import gradio as gr

from dreambooth import shared
from dreambooth.shared import (
    get_launch_errors,
)
from dreambooth.utils.model_utils import (
    get_ft_models,
)
from dreambooth.utils.ui_utils import ElementManager
from finetune.ft_ui_functions import create_ft_workspace
from helpers.log_parser import LogParser
from modules import script_callbacks, sd_models
from modules.ui import gr_show, create_refresh_button
from preprocess.preprocess_utils import load_image_caption, check_preprocess_path

preprocess_params = []
params_to_save = []
params_to_load = []
refresh_symbol = "\U0001f504"  # 🔄
delete_symbol = "\U0001F5D1"  # 🗑️
update_symbol = "\U0001F51D"  # 🠝
log_parser = LogParser()
show_advanced = True

setting_elements_list = []
setting_elements_dict = {}


def read_metadata_from_safetensors(filename):
    with open(filename, mode="rb") as file:
        # Read metadata length
        metadata_len = int.from_bytes(file.read(8), "little")

        # Read the metadata based on its length
        json_data = file.read(metadata_len).decode('utf-8')

        res = {}

        # Check if it's a valid JSON string
        try:
            json_obj = json.loads(json_data)
        except json.JSONDecodeError:
            return res

        # Extract metadata
        metadata = json_obj.get("__metadata__", {})
        if not isinstance(metadata, dict):
            return res

        # Process the metadata to handle nested JSON strings
        for k, v in metadata.items():
            # if not isinstance(v, str):
            #     raise ValueError("All values in __metadata__ must be strings")

            # If the string value looks like a JSON string, attempt to parse it
            if v.startswith('{'):
                try:
                    res[k] = json.loads(v)
                except Exception:
                    res[k] = v
            else:
                res[k] = v

        return res


def get_sd_models():
    sd_models.list_models()
    sd_list = sd_models.checkpoints_list
    names = []
    for key in sd_list:
        names.append(key)
    return names


def list_training_types():
    # Enumerate the files in dreambooth.training
    training_types = []
    for file in os.listdir(
            os.path.join(shared.script_path, "extensions", "sd_dreambooth_extension", "finetune", "scripts")):
        if file.endswith(".py") and not file.startswith("__"):
            training_types.append(file[:-3])
    print(f"Training types: {training_types}")
    return training_types


def get_training_ui_script(training_type):
    module_path = ["finetune", "ui", f"{training_type}_ui"]
    print(f"Getting training ui script: {module_path}")
    return ".".join(module_path)


def get_training_config_script(training_type):
    module_path = ["finetune", "configs", f"{training_type}_config"]
    return ".".join(module_path)


def select_training_params(training_type):
    global setting_elements_list
    global setting_elements_dict
    manager = ElementManager()
    elements = manager.get_elements(training_type)
    setting_elements_list = []
    setting_elements_dict = elements
    for key in elements:
        setting_elements_list.append(elements[key])
    return elements


def load_workspace(workspace_name):
    if workspace_name == "":
        return
    workspace_config = {}
    workspace_path = os.path.join(shared.models_path, "finetune", workspace_name)
    workspace_config_path = os.path.join(workspace_path, "config.json")
    if os.path.exists(workspace_config_path):
        with open(workspace_config_path, "r") as f:
            workspace_config = json.load(f)


def get_training_settings(workspace_name, training_type):
    workspace_value = workspace_name.value
    settings = {}

    if not workspace_value:
        existing_settings = {}
    else:
        workspace_path = os.path.join(shared.models_path, "finetune", workspace_value)
        settings_path = os.path.join(workspace_path, f"settings_{training_type}.json")
        if os.path.exists(settings_path):
            with open(settings_path, "r") as f:
                existing_settings = json.load(f)
        else:
            existing_settings = {}
    base_settings_class = get_training_config_script(training_type)
    base_settings_module = importlib.import_module(base_settings_class)
    setting_class_name = base_settings_class.split(".")[-1].replace("_", " ").title().replace(" ", "")
    settings_class = getattr(base_settings_module, setting_class_name)

    # Instantiate the settings class
    settings_instance = settings_class()

    # Now you can use the to_dict() method on this instance
    base_dict = settings_instance.to_dict()
    for key in base_dict:
        if not key.startswith("__"):
            settings[key] = base_dict[key]
    for key in existing_settings:
        settings[key] = existing_settings[key]

    return settings


def load_ft_params(workspace_name, training_type):
    """Load the parameters for a given workspace and training type
    Returns:
        ft_workspace_name,
        ft_model_path,
        ft_has_ema,
        ft_model_type,
        ft_status,
        setting_elements_list
    """
    global setting_elements_dict
    global setting_elements_list
    workspace_params = get_training_settings(workspace_name, training_type)
    setting_elements_list = []

    for key in workspace_params:
        if key in setting_elements_dict:
            setting_elements_dict[key].value = workspace_params[key]
            setting_elements_list.append(setting_elements_dict[key])
    return workspace_name, workspace_params["model_path"], workspace_params["has_ema"], workspace_params[
        "model_type"], "", setting_elements_list


def save_ft_params(workspace_name, training_type):
    """Save the parameters for a given workspace and training type
    Returns:
        ft_workspace_name,
        ft_model_path,
        ft_has_ema,
        ft_model_type,
        ft_status,
        setting_elements_list
    """
    global setting_elements_list
    workspace_path = os.path.join(shared.models_path, "finetune", workspace_name)
    settings_path = os.path.join(workspace_path, f"settings_{training_type}.json")
    settings = {}
    for element in setting_elements_list:
        if hasattr(element, "value"):
            settings[element.elem_id] = element.value
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)
    outputs = [workspace_name, settings["model_path"], settings["has_ema"], settings["model_type"], ""]
    for element in setting_elements_list:
        outputs.append(element)


def start_ft_train(workspace_name, training_type, elements_list):
    """Start training a given workspace and training type
    Returns:
        ft_gallery,
        ft_status
    """
    global setting_elements_list

    workspace_path = os.path.join(shared.models_path, "finetune", workspace_name)
    settings_path = os.path.join(workspace_path, f"settings_{training_type}.json")
    settings = {}
    for element in setting_elements_list:
        if hasattr(element, "value"):
            settings[element.elem_id] = element.value
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)
    outputs = [workspace_name, settings["model_path"], settings["has_ema"], settings["model_type"], ""]
    for element in setting_elements_list:
        outputs.append(element)
    return outputs


def new_ui_tabs():
    global setting_elements_list
    global setting_elements_dict

    with gr.Blocks() as finetune_interface:
        # region Top Row
        with gr.Row(equal_height=True, elem_id="FtTopRow"):
            ft_load_params = gr.Button(value="Load Settings", elem_id="ft_load_params", size="sm")
            ft_save_params = gr.Button(value="Save Settings", elem_id="ft_save_config", size="sm")
            ft_train_model = gr.Button(
                value="Train", variant="primary", elem_id="ft_train", size="sm"
            )
            ft_cancel = gr.Button(value="Cancel", elem_id="ft_cancel", size="sm", visible=False)
        # endregion

        # region Alert Row
        with gr.Row():
            gr.HTML(value="Select or create a workspace to begin.", elem_id="hint_row")
        # endregion

        # region Workspace Detail Row
        with gr.Row(elem_id="WorkspaceDetailRow", visible=False, variant="compact") as ft_model_info:
            with gr.Column():
                with gr.Row(variant="compact"):
                    with gr.Column():
                        with gr.Row(variant="compact"):
                            gr.HTML(value="Loaded Model:")
                            ft_model_path = gr.HTML()
                        with gr.Row(variant="compact"):
                            gr.HTML(value="Source Checkpoint:")
                            ft_src = gr.HTML()
                    # TODO: Auto-load selected training type params from model section here
                    with gr.Column():
                        with gr.Row(variant="compact"):
                            gr.HTML(value="Model Epoch:")
                            ft_epochs = gr.HTML(elem_id="ft_epochs")
                        with gr.Row(variant="compact"):
                            gr.HTML(value="Model Revision:")
                            ft_revision = gr.HTML(elem_id="ft_revision")
                    with gr.Column():
                        with gr.Row(variant="compact"):
                            gr.HTML(value="Model type:")
                            ft_model_type = gr.HTML(elem_id="ft_model_type")
                        with gr.Row(variant="compact"):
                            gr.HTML(value="Has EMA:")
                            ft_has_ema = gr.HTML(elem_id="ft_has_ema")
        # endregion

        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", elem_id="SettingsPanel"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=100, elem_classes="halfElement"):
                        gr.HTML(value="<span class='hh'>Settings</span>")
                    with gr.Column(scale=1, min_width=100, elem_classes="halfElement"):
                        ft_show_advanced = gr.Button(value="Show Advanced", size="sm", elem_classes="advBtn",
                                                     visible=False)
                        ft_hide_advanced = gr.Button(value="Hide Advanced", variant="primary", size="sm",
                                                     elem_id="ft_hide_advanced", elem_classes="advBtn")

                # region Workspace Tab
                with gr.Tab("Workspace", elem_id="WorkspacePanel") as workspace_tab:
                    with gr.Column():
                        with gr.Tab("Select"):
                            with gr.Row():
                                ft_workspace_name = gr.Dropdown(
                                    label="Model", choices=sorted(get_ft_models())
                                )
                                create_refresh_button(
                                    ft_workspace_name,
                                    get_ft_models,
                                    lambda: {"choices": sorted(get_ft_models())},
                                    "refresh_ft_models",
                                )
                        with gr.Tab("Create"):
                            with gr.Column():
                                ft_create_workspace = gr.Button(
                                    value="Create Workspace", variant="primary"
                                )
                            ft_new_workspace_name = gr.Textbox(label="Name")
                            with gr.Row():
                                ft_create_from_hub = gr.Checkbox(
                                    label="Create From Hub", value=False
                                )
                                ft_model_type_select = gr.Dropdown(label="Model Type",
                                                                   choices=["v1x", "v2x-512", "v2x", "sdxl",
                                                                            "ControlNet"], value="v1x")
                            with gr.Column(visible=False) as hub_row:
                                ft_new_model_url = gr.Textbox(
                                    label="Model Path",
                                    placeholder="runwayml/stable-diffusion-v1-5",
                                )
                                ft_new_model_token = gr.Textbox(
                                    label="HuggingFace Token", value=""
                                )
                            with gr.Column(visible=True) as local_row:
                                with gr.Row():
                                    ft_new_model_src = gr.Dropdown(
                                        label="Source Checkpoint",
                                        choices=sorted(get_sd_models()),
                                    )
                                    create_refresh_button(
                                        ft_new_model_src,
                                        get_sd_models,
                                        lambda: {"choices": sorted(get_sd_models())},
                                        "refresh_sd_models",
                                    )
                    with gr.Column():
                        with gr.Accordion(open=False, label="Resources"):
                            with gr.Column():
                                gr.HTML(
                                    value="<a class=\"hyperlink\" href=\"https://github.com/d8ahazard/sd_dreambooth_extension/wiki/ELI5-Training\">Beginners guide</a>",
                                )
                                gr.HTML(
                                    value="<a class=\"hyperlink\" href=\"https://github.com/d8ahazard/sd_dreambooth_extension/releases/latest\">Release notes</a>",
                                )
                # endregion

                # region Preprocess Tab
                with gr.Tab("Preprocess", elem_id="PreprocessPanel") as preprocess_tab:
                    with gr.Row():
                        with gr.Column(scale=2, variant="compact"):
                            ft_preprocess_path = gr.Textbox(
                                label="Image Path", value="", placeholder="Enter the path to your images"
                            )
                        with gr.Column(variant="compact"):
                            ft_preprocess_recursive = gr.Checkbox(
                                label="Recursive", value=False, container=True, elem_classes=["singleCheckbox"]
                            )
                    with gr.Row():
                        with gr.Tab("Auto-Caption"):
                            with gr.Row():
                                gr.HTML(value="Auto-Caption")
                        with gr.Tab("Edit Captions"):
                            with gr.Row():
                                ft_preprocess_autosave = gr.Checkbox(
                                    label="Autosave", value=False
                                )
                            with gr.Row():
                                gr.HTML(value="Edit Captions")
                        with gr.Tab("Edit Images"):
                            with gr.Row():
                                gr.HTML(value="Edit Images")
                    with gr.Row():
                        ft_preprocess = gr.Button(
                            value="Preprocess", variant="primary"
                        )
                        ft_preprocess_all = gr.Button(
                            value="Preprocess All", variant="primary"
                        )
                    with gr.Row():
                        ft_preprocess_all = gr.Button(
                            value="Preprocess All", variant="primary"
                        )
                # endregion

                # region Settings Tab
                with gr.Tab("Settings", elem_id="TabNU") as nu_tab:
                    training_types = list_training_types()
                    # Insert a blank option at the beginning
                    training_types.insert(0, "")
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            ft_training_type = gr.Dropdown(
                                label="Training Type",
                                value="",
                                choices=training_types,
                            )
                        with gr.Row():
                            training_uis = []
                            for tt in training_types:
                                if tt == "":
                                    continue
                                ui_script = get_training_ui_script(tt)
                                # Import the ui script and call its render() function
                                if ui_script is not None:
                                    ui_module = importlib.import_module(ui_script)
                                    with gr.Column(visible=False, elem_id=f"{tt}_ui") as some_col:
                                        ui_module.render()
                                        training_uis.append(some_col)

                            def update_visibility(training_type):
                                out_elements = []
                                global setting_elements_list
                                global setting_elements_dict
                                print(f"Updating visibility for {training_type}")
                                for col in training_uis:
                                    # Check if the elem_id of the column matches the selected training type
                                    is_visible = col.elem_id == f"{training_type}_ui"
                                    out_elements.append(gr.update(visible=is_visible))
                                setting_elements_dict = get_training_settings(ft_workspace_name, training_type)
                                setting_elements_list = []
                                for key in setting_elements_dict:
                                    setting_elements_list.append(setting_elements_dict[key])
                                return out_elements

                            ft_training_type.change(
                                fn=update_visibility,
                                inputs=[ft_training_type],
                                outputs=training_uis
                            )
                # endregion

            # region Preprocess View
            with gr.Column(variant="panel", visible=False) as ft_preprocess_view:
                with gr.Row():
                    ft_pp_select_all = gr.Button(value="Add All", size="sm")
                    ft_pp_select_current = gr.Button(value="Add Selection", size="sm")
                    ft_pp_clear_current = gr.Button(value="Remove Selection", size="sm")
                    ft_pp_clear_all = gr.Button(value="Remove All", size="sm")
                ft_preprocess_status = gr.HTML()
                ft_preprocess_gallery = gr.Gallery(
                    label="Preprocess", show_label=False, elem_id="ft_preprocess_gallery", columns=4, rows=4,
                    preview=False
                )
                ft_preprocess_gallery_2 = gr.Gallery(
                    label="Preprocess2", show_label=False, elem_id="ft_preprocess_gallery_2", columns=4, rows=4,
                    preview=False
                )
            # endregion

            # region Status View
            with gr.Column(variant="panel") as ft_output_view:
                with gr.Row():
                    with gr.Column(scale=1, min_width=110):
                        gr.HTML(value="<span class='hh'>Output</span>")
                    with gr.Column(scale=1, min_width=110):
                        ft_check_progress_initial = gr.Button(
                            value=update_symbol,
                            elem_id="ft_check_progress_initial",
                            visible=False,
                        )
                        # These two should be updated while doing things
                        ft_active = gr.Checkbox(elem_id="ft_active", value=False, visible=False)

                        ui_check_progress_initial = gr.Button(
                            value="Refresh", elem_id="ui_check_progress_initial", elem_classes="advBtn", size="sm"
                        )
                ft_status = gr.HTML(elem_id="ft_status", value="")
                ft_progressbar = gr.HTML(elem_id="ft_progressbar")
                ft_gallery = gr.Gallery(
                    label="Output", show_label=False, elem_id="ft_gallery", columns=4, rows=4, preview=False
                )
                ft_preview = gr.Image(elem_id="ft_preview", visible=False)
                ft_prompt_list = gr.HTML(
                    elem_id="ft_prompt_list", value="", visible=False
                )
                ft_gallery_prompt = gr.HTML(elem_id="ft_gallery_prompt", value="")
                ft_check_progress = gr.Button(
                    "Check Progress", elem_id=f"ft_check_progress", visible=False
                )
                ft_update_params = gr.Button(
                    "Update Parameters", elem_id="ft_update_params", visible=False
                )
                ft_launch_error = gr.HTML(
                    elem_id="launch_errors", visible=False, value=get_launch_errors
                )
            # endregion

        input_elements = [
            ft_workspace_name,
            ft_training_type
        ]

        # region Top Row Listeners
        output_elements = [
            ft_workspace_name,
            ft_model_path,
            ft_has_ema,
            ft_model_type,
            ft_status
        ]
        for element in setting_elements_list:
            print(f"Adding {element} to output elements")
            output_elements.append(element)

        ft_load_params.click(
            fn=load_ft_params,
            inputs=[ft_workspace_name, ft_training_type],
            outputs=output_elements,
        )

        ft_save_params.click(
            fn=save_ft_params,
            inputs=[ft_workspace_name, ft_training_type],
            outputs=output_elements,
        )

        ft_train_model.click(
            fn=start_ft_train,
            inputs=[ft_workspace_name, ft_training_type],
            outputs=output_elements,
        )
        # endregion

        # region Workspace Listeners

        # When a workspace is selected, load the settings for that workspace
        ft_workspace_name.change(
            fn=load_workspace,
            inputs=[ft_workspace_name],
            outputs=setting_elements_list,
        )

        ft_create_from_hub.change(
            fn=lambda x: {hub_row: gr_show(x is True), local_row: gr_show(x is False)},
            inputs=[ft_create_from_hub],
            outputs=[hub_row, local_row],
        )

        new_workspace_params = [
            ft_new_workspace_name,
            ft_create_from_hub,
            ft_new_model_url,
            ft_new_model_token,
            ft_new_model_src,
            ft_model_type_select
        ]

        ft_create_workspace.click(
            fn=create_ft_workspace,
            inputs=new_workspace_params,
            outputs=[
                ft_workspace_name,
                ft_model_path,
                ft_has_ema,
                ft_model_type,
                ft_status,
            ],
        )
        # endregion

        # region Preprocessing Listeners
        global preprocess_params

        preprocess_params = [
            ft_preprocess_path,
            ft_preprocess_recursive
        ]

        ft_preprocess_path.change(
            fn=check_preprocess_path,
            inputs=[ft_preprocess_path, ft_preprocess_recursive],
            outputs=[ft_preprocess_status, ft_preprocess_gallery],
        )

        ft_preprocess_gallery.select(load_image_caption, None, ft_preprocess_status)
        # endregion
    return ((finetune_interface, "FineTune+", "vt_v2"),)


script_callbacks.on_ui_tabs(new_ui_tabs)
