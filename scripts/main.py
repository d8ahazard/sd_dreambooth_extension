import importlib
import json
import time
from typing import List

import gradio as gr

from extensions.sd_dreambooth_extension.dreambooth.dataclasses.db_config import from_file, save_config
from extensions.sd_dreambooth_extension.dreambooth.diff_to_sd import compile_checkpoint
from extensions.sd_dreambooth_extension.dreambooth.diff_to_sdxl import compile_checkpoint as compile_checkpoint_sdxl
from extensions.sd_dreambooth_extension.dreambooth.secret import (
    get_secret,
    create_secret,
    clear_secret,
)
from extensions.sd_dreambooth_extension.dreambooth.shared import (
    status,
    get_launch_errors,
)
from extensions.sd_dreambooth_extension.dreambooth.ui_functions import (
    performance_wizard,
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
from extensions.sd_dreambooth_extension.dreambooth.utils.image_utils import (
    get_scheduler_names,
)
from extensions.sd_dreambooth_extension.dreambooth.utils.model_utils import (
    get_db_models,
    get_sorted_lora_models,
    get_model_snapshots,
    get_shared_models,
)
from extensions.sd_dreambooth_extension.dreambooth.utils.utils import (
    list_attention,
    list_precisions,
    wrap_gpu_call,
    printm,
    list_optimizer,
    list_schedulers, select_precision, select_attention,
)
from extensions.sd_dreambooth_extension.dreambooth.webhook import save_and_test_webhook
from helpers.log_parser import LogParser
from helpers.version_helper import check_updates
from modules import script_callbacks, sd_models
from modules.ui import gr_show, create_refresh_button
from preprocess.preprocess_utils import load_image_caption

preprocess_params = []
params_to_save = []
params_to_load = []
refresh_symbol = "\U0001f504"  # 🔄
delete_symbol = "\U0001F5D1"  # 🗑️
update_symbol = "\U0001F51D"  # 🠝

log_parser = LogParser()
show_advanced = True


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
    prompts = status.sample_prompts
    preview = None
    gallery = None

    if image is None:
        preview = gr.update(visible=False, value=None)
        gallery = gr.update(visible=True)
    else:
        if isinstance(image, List):
            samples = image
            if isinstance(prompts, List) and len(prompts) == len(image):
                samples = []
                for i in range(len(image)):
                    samples.append((image[i], prompts[i]))
            if len(image) >= 1:
                status.current_image = None
                preview = gr.update(visible=False, value=None)
                gallery = gr.update(visible=True, value=samples)
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
    if config.model_type == "SDXL":
        res = compile_checkpoint_sdxl(model_name, lora_path, True, False, config.snapshot)
    else:
        res = compile_checkpoint(model_name, lora_path, True, True, config.snapshot)
    return res


def on_ui_tabs():
    with gr.Blocks() as dreambooth_interface:
        # Top button row
        with gr.Row(equal_height=True, elem_id="DbTopRow"):
            db_load_params = gr.Button(value="Load Settings", elem_id="db_load_params", size="sm")
            db_save_params = gr.Button(value="Save Settings", elem_id="db_save_config", size="sm")
            db_train_model = gr.Button(
                value="Train", variant="primary", elem_id="db_train", size="sm"
            )
            db_generate_checkpoint = gr.Button(
                value="Generate Ckpt", elem_id="db_gen_ckpt", size="sm"
            )
            db_generate_checkpoint_during = gr.Button(
                value="Save Weights", elem_id="db_gen_ckpt_during", size="sm"
            )
            db_train_sample = gr.Button(
                value="Generate Samples", elem_id="db_train_sample", size="sm"
            )
            db_cancel = gr.Button(value="Cancel", elem_id="db_cancel", size="sm")
        with gr.Row():
            gr.HTML(value="Select or create a model to begin.", elem_id="hint_row")
        with gr.Row(elem_id="ModelDetailRow", visible=False, variant="compact") as db_model_info:
            with gr.Column():
                with gr.Row(variant="compact"):
                    with gr.Column():
                        with gr.Row(variant="compact"):
                            gr.HTML(value="Loaded Model:")
                            db_model_path = gr.HTML()
                        with gr.Row(variant="compact"):
                            gr.HTML(value="Source Checkpoint:")
                            db_src = gr.HTML()
                    with gr.Column():
                        with gr.Row(variant="compact"):
                            gr.HTML(value="Model Epoch:")
                            db_epochs = gr.HTML(elem_id="db_epochs")
                        with gr.Row(variant="compact"):
                            gr.HTML(value="Model Revision:")
                            db_revision = gr.HTML(elem_id="db_revision")
                    with gr.Column():
                        with gr.Row(variant="compact"):
                            gr.HTML(value="Model type:")
                            db_model_type = gr.HTML(elem_id="db_model_type")
                        with gr.Row(variant="compact"):
                            gr.HTML(value="Has EMA:")
                            db_has_ema = gr.HTML(elem_id="db_has_ema")
                    with gr.Row(variant="compact", visible=False):
                        gr.HTML(value="Experimental Shared Source:")
                        db_shared_diffusers_path = gr.HTML()
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", elem_id="SettingsPanel"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=100, elem_classes="halfElement"):
                        gr.HTML(value="<span class='hh'>Settings</span>")
                    with gr.Column(scale=1, min_width=100, elem_classes="halfElement"):
                        db_show_advanced = gr.Button(value="Show Advanced", size="sm", elem_classes="advBtn",
                                                     visible=False)
                        db_hide_advanced = gr.Button(value="Hide Advanced", variant="primary", size="sm",
                                                     elem_id="db_hide_advanced", elem_classes="advBtn")
                with gr.Tab("Model", elem_id="ModelPanel"):
                    with gr.Column():
                        with gr.Tab("Select"):
                            with gr.Row():
                                db_model_name = gr.Dropdown(
                                    label="Model", choices=sorted(get_db_models())
                                )
                                create_refresh_button(
                                    db_model_name,
                                    get_db_models,
                                    lambda: {"choices": sorted(get_db_models())},
                                    "refresh_db_models",
                                )
                            with gr.Row() as db_snapshot_row:
                                db_snapshot = gr.Dropdown(
                                    label="Snapshot to Resume",
                                    choices=sorted(get_model_snapshots()),
                                )
                                create_refresh_button(
                                    db_snapshot,
                                    get_model_snapshots,
                                    lambda: {"choices": sorted(get_model_snapshots())},
                                    "refresh_db_snapshots",
                                )
                            with gr.Row(visible=False) as lora_model_row:
                                db_lora_model_name = gr.Dropdown(
                                    label="Lora Model", choices=get_sorted_lora_models()
                                )
                                create_refresh_button(
                                    db_lora_model_name,
                                    get_sorted_lora_models,
                                    lambda: {"choices": get_sorted_lora_models()},
                                    "refresh_lora_models",
                                )
                        with gr.Tab("Create"):
                            with gr.Column():
                                db_create_model = gr.Button(
                                    value="Create Model", variant="primary"
                                )
                            db_new_model_name = gr.Textbox(label="Name")
                            with gr.Row():
                                db_create_from_hub = gr.Checkbox(
                                    label="Create From Hub", value=False
                                )
                                db_model_type_select = gr.Dropdown(label="Model Type",
                                                                   choices=["v1x", "v2x-512", "v2x", "SDXL",
                                                                            "ControlNet"], value="v1x")
                                db_use_shared_src = gr.Checkbox(
                                    label="Experimental Shared Src", value=False, visible=False
                                )
                            with gr.Column(visible=False) as hub_row:
                                db_new_model_url = gr.Textbox(
                                    label="Model Path",
                                    placeholder="runwayml/stable-diffusion-v1-5",
                                )
                                db_new_model_token = gr.Textbox(
                                    label="HuggingFace Token", value=""
                                )
                            with gr.Column(visible=True) as local_row:
                                with gr.Row():
                                    db_new_model_src = gr.Dropdown(
                                        label="Source Checkpoint",
                                        choices=sorted(get_sd_models()),
                                    )
                                    create_refresh_button(
                                        db_new_model_src,
                                        get_sd_models,
                                        lambda: {"choices": sorted(get_sd_models())},
                                        "refresh_sd_models",
                                    )
                            with gr.Column(visible=False) as shared_row:
                                with gr.Row():
                                    db_new_model_shared_src = gr.Dropdown(
                                        label="EXPERIMENTAL: LoRA Shared Diffusers Source",
                                        choices=sorted(get_shared_models()),
                                        value="",
                                        visible=False
                                    )
                                    create_refresh_button(
                                        db_new_model_shared_src,
                                        get_shared_models,
                                        lambda: {"choices": sorted(get_shared_models())},
                                        "refresh_shared_models",
                                    )
                            db_new_model_extract_ema = gr.Checkbox(
                                label="Extract EMA Weights", value=False
                            )
                            db_train_unfrozen = gr.Checkbox(label="Unfreeze Model", value=True)
                    with gr.Column():
                        with gr.Accordion(open=False, label="Resources"):
                            with gr.Column():
                                gr.HTML(
                                    value="<a class=\"hyperlink\" href=\"https://github.com/d8ahazard/sd_dreambooth_extension/wiki/ELI5-Training\">Beginners guide</a>",
                                )
                                gr.HTML(
                                    value="<a class=\"hyperlink\" href=\"https://github.com/d8ahazard/sd_dreambooth_extension/releases/latest\">Release notes</a>",
                                )
                # with gr.Tab("Preprocess", elem_id="PreprocessPanel", visible=False):
                #     with gr.Row():
                #         with gr.Column(scale=2, variant="compact"):
                #             db_preprocess_path = gr.Textbox(
                #                 label="Image Path", value="", placeholder="Enter the path to your images"
                #             )
                #         with gr.Column(variant="compact"):
                #             db_preprocess_recursive = gr.Checkbox(
                #                 label="Recursive", value=False, container=True, elem_classes=["singleCheckbox"]
                #             )
                #     with gr.Row():
                #         with gr.Tab("Auto-Caption"):
                #             with gr.Row():
                #                 gr.HTML(value="Auto-Caption")
                #         with gr.Tab("Edit Captions"):
                #             with gr.Row():
                #                 db_preprocess_autosave = gr.Checkbox(
                #                     label="Autosave", value=False
                #                 )
                #             with gr.Row():
                #                 gr.HTML(value="Edit Captions")
                #         with gr.Tab("Edit Images"):
                #             with gr.Row():
                #                 gr.HTML(value="Edit Images")
                #     with gr.Row():
                #         db_preprocess = gr.Button(
                #             value="Preprocess", variant="primary"
                #         )
                #         db_preprocess_all = gr.Button(
                #             value="Preprocess All", variant="primary"
                #         )
                #     with gr.Row():
                #         db_preprocess_all = gr.Button(
                #             value="Preprocess All", variant="primary"
                #         )
                with gr.Tab("Concepts", elem_id="TabConcepts") as concept_tab:
                    with gr.Column(variant="panel"):
                        with gr.Accordion(open=False, label="Concept 1"):
                            (
                                c1_instance_data_dir,
                                c1_class_data_dir,
                                c1_instance_prompt,
                                c1_class_prompt,
                                c1_save_sample_prompt,
                                c1_save_sample_template,
                                c1_instance_token,
                                c1_class_token,
                                c1_num_class_images_per,
                                c1_class_negative_prompt,
                                c1_class_guidance_scale,
                                c1_class_infer_steps,
                                c1_save_sample_negative_prompt,
                                c1_n_save_sample,
                                c1_sample_seed,
                                c1_save_guidance_scale,
                                c1_save_infer_steps,
                            ) = build_concept_panel(1)

                        with gr.Accordion(open=False, label="Concept 2"):
                            (
                                c2_instance_data_dir,
                                c2_class_data_dir,
                                c2_instance_prompt,
                                c2_class_prompt,
                                c2_save_sample_prompt,
                                c2_save_sample_template,
                                c2_instance_token,
                                c2_class_token,
                                c2_num_class_images_per,
                                c2_class_negative_prompt,
                                c2_class_guidance_scale,
                                c2_class_infer_steps,
                                c2_save_sample_negative_prompt,
                                c2_n_save_sample,
                                c2_sample_seed,
                                c2_save_guidance_scale,
                                c2_save_infer_steps,
                            ) = build_concept_panel(2)

                        with gr.Accordion(open=False, label="Concept 3"):
                            (
                                c3_instance_data_dir,
                                c3_class_data_dir,
                                c3_instance_prompt,
                                c3_class_prompt,
                                c3_save_sample_prompt,
                                c3_save_sample_template,
                                c3_instance_token,
                                c3_class_token,
                                c3_num_class_images_per,
                                c3_class_negative_prompt,
                                c3_class_guidance_scale,
                                c3_class_infer_steps,
                                c3_save_sample_negative_prompt,
                                c3_n_save_sample,
                                c3_sample_seed,
                                c3_save_guidance_scale,
                                c3_save_infer_steps,
                            ) = build_concept_panel(3)

                        with gr.Accordion(open=False, label="Concept 4"):
                            (
                                c4_instance_data_dir,
                                c4_class_data_dir,
                                c4_instance_prompt,
                                c4_class_prompt,
                                c4_save_sample_prompt,
                                c4_save_sample_template,
                                c4_instance_token,
                                c4_class_token,
                                c4_num_class_images_per,
                                c4_class_negative_prompt,
                                c4_class_guidance_scale,
                                c4_class_infer_steps,
                                c4_save_sample_negative_prompt,
                                c4_n_save_sample,
                                c4_sample_seed,
                                c4_save_guidance_scale,
                                c4_save_infer_steps,
                            ) = build_concept_panel(4)
                with gr.Tab("Parameters", elem_id="TabSettings"):
                    db_performance_wizard = gr.Button(value="Performance Wizard (WIP)", visible=False)
                    with gr.Accordion(open=False, label="Performance"):
                        db_use_ema = gr.Checkbox(
                            label="Use EMA", value=False
                        )
                        db_optimizer = gr.Dropdown(
                            label="Optimizer",
                            value="8bit AdamW",
                            choices=list_optimizer(),
                        )
                        db_mixed_precision = gr.Dropdown(
                            label="Mixed Precision",
                            value=select_precision(),
                            choices=list_precisions(),
                        )
                        db_full_mixed_precision = gr.Checkbox(
                            label="Full Mixed Precision", value=True
                        )
                        db_attention = gr.Dropdown(
                            label="Memory Attention",
                            value=select_attention(),
                            choices=list_attention(),
                        )
                        db_cache_latents = gr.Checkbox(
                            label="Cache Latents", value=True
                        )
                        db_train_unet = gr.Checkbox(
                            label="Train UNET", value=True
                        )
                        db_stop_text_encoder = gr.Slider(
                            label="Step Ratio of Text Encoder Training",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=1.0,
                            visible=True,
                        )
                        db_offset_noise = gr.Slider(
                            label="Offset Noise",
                            minimum=-1,
                            maximum=1,
                            step=0.01,
                            value=0,
                        )
                        db_freeze_clip_normalization = gr.Checkbox(
                            label="Freeze CLIP Normalization Layers",
                            visible=True,
                            value=False,
                        )
                        db_clip_skip = gr.Slider(
                            label="Clip Skip",
                            value=2,
                            minimum=1,
                            maximum=12,
                            step=1,
                        )
                        db_weight_decay = gr.Slider(
                            label="Weight Decay",
                            minimum=0,
                            maximum=1,
                            step=0.001,
                            value=0.01,
                            visible=True,
                        )
                        db_tenc_weight_decay = gr.Slider(
                            label="TENC Weight Decay",
                            minimum=0,
                            maximum=1,
                            step=0.001,
                            value=0.01,
                            visible=True,
                        )
                        db_tenc_grad_clip_norm = gr.Slider(
                            label="TENC Gradient Clip Norm",
                            minimum=0,
                            maximum=128,
                            step=0.25,
                            value=0,
                            visible=True,
                        )
                        db_min_snr_gamma = gr.Slider(
                            label="Min SNR Gamma",
                            minimum=0,
                            maximum=10,
                            step=0.1,
                            visible=True,
                        )
                        db_use_dream = gr.Checkbox(
                            label="Use DREAM", value=False
                        )
                        db_dream_detail_preservation = gr.Slider(
                            label="DREAM detail preservation",
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.5,
                            visible=True,
                        )
                        db_freeze_spectral_norm = gr.Checkbox(
                            label="Freeze Spectral Norm", value=False
                        )
                        db_pad_tokens = gr.Checkbox(
                            label="Pad Tokens", value=True
                        )
                        db_strict_tokens = gr.Checkbox(
                            label="Strict Tokens", value=False
                        )
                        db_shuffle_tags = gr.Checkbox(
                            label="Shuffle Tags", value=True
                        )
                        db_max_token_length = gr.Slider(
                            label="Max Token Length",
                            minimum=75,
                            maximum=300,
                            step=75,
                        )
                    with gr.Accordion(open=False, label="Intervals"):
                        db_num_train_epochs = gr.Slider(
                            label="Training Steps Per Image (Epochs)",
                            value=100,
                            maximum=1000,
                            step=1,
                        )
                        db_epoch_pause_frequency = gr.Slider(
                            label="Pause After N Epochs",
                            value=0,
                            maximum=100,
                            step=1,
                        )
                        db_epoch_pause_time = gr.Slider(
                            label="Amount of time to pause between Epochs (s)",
                            value=0,
                            maximum=3600,
                            step=1,
                        )
                        db_save_embedding_every = gr.Slider(
                            label="Save Model Frequency (Epochs)",
                            value=25,
                            maximum=1000,
                            step=1,
                        )
                        db_save_preview_every = gr.Slider(
                            label="Save Preview(s) Frequency (Epochs)",
                            value=5,
                            maximum=1000,
                            step=1,
                        )
                    with gr.Accordion(open=False, label="Batch Sizes") as db_batch_size_view:
                        db_train_batch_size = gr.Slider(
                            label="Batch Size",
                            value=1,
                            minimum=1,
                            maximum=100,
                            step=1,
                        )
                        db_gradient_accumulation_steps = gr.Slider(
                            label="Gradient Accumulation Steps",
                            value=1,
                            minimum=1,
                            maximum=100,
                            step=1,
                        )
                        db_sample_batch_size = gr.Slider(
                            label="Class Batch Size",
                            minimum=1,
                            maximum=100,
                            value=1,
                            step=1,
                        )
                        db_gradient_set_to_none = gr.Checkbox(
                            label="Set Gradients to None When Zeroing", value=True
                        )
                        db_gradient_checkpointing = gr.Checkbox(
                            label="Gradient Checkpointing", value=True
                        )
                    with gr.Accordion(open=False, label="Learning Rate"):
                        with gr.Row(visible=False) as lora_lr_row:
                            db_lora_learning_rate = gr.Number(
                                label="Lora UNET Learning Rate", value=1e-4
                            )
                            db_lora_txt_learning_rate = gr.Number(
                                label="Lora Text Encoder Learning Rate", value=5e-5
                            )
                        with gr.Row() as standard_lr_row:
                            db_learning_rate = gr.Number(
                                label="Learning Rate", value=2e-6
                            )
                            db_txt_learning_rate = gr.Number(
                                label="Text Encoder Learning Rate", value=1e-6
                            )

                        db_lr_scheduler = gr.Dropdown(
                            label="Learning Rate Scheduler",
                            value="constant_with_warmup",
                            choices=list_schedulers(),
                        )
                        db_learning_rate_min = gr.Number(
                            label="Min Learning Rate", value=1e-6, visible=False
                        )
                        db_lr_cycles = gr.Number(
                            label="Number of Hard Resets",
                            value=1,
                            precision=0,
                            visible=False,
                        )
                        db_lr_factor = gr.Number(
                            label="Constant/Linear Starting Factor",
                            value=0.5,
                            precision=2,
                            visible=False,
                        )
                        db_lr_power = gr.Number(
                            label="Polynomial Power",
                            value=1.0,
                            precision=1,
                            visible=False,
                        )
                        db_lr_scale_pos = gr.Slider(
                            label="Scale Position",
                            value=0.5,
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            visible=False,
                        )
                        db_lr_warmup_steps = gr.Slider(
                            label="Learning Rate Warmup Steps",
                            value=500,
                            step=5,
                            maximum=1000,
                        )
                    with gr.Accordion(open=False, label="Lora"):
                        db_use_lora = gr.Checkbox(label="Use LORA", value=False)
                        db_use_lora_extended = gr.Checkbox(
                            label="Use Lora Extended",
                            value=False,
                            visible=False,
                        )
                        db_train_imagic = gr.Checkbox(label="Train Imagic Only", value=False, visible=False)
                        db_train_inpainting = gr.Checkbox(
                            label="Train Inpainting Model",
                            value=False,
                            visible=False,
                        )
                        with gr.Column(visible=False) as lora_rank_col:
                            db_lora_unet_rank = gr.Slider(
                                label="Lora UNET Rank",
                                value=4,
                                minimum=2,
                                maximum=128,
                                step=2,
                            )
                            db_lora_txt_rank = gr.Slider(
                                label="Lora Text Encoder Rank",
                                value=4,
                                minimum=2,
                                maximum=128,
                                step=2,
                            )
                            db_lora_weight = gr.Slider(
                                label="Lora Weight (Alpha)",
                                value=0.8,
                                minimum=0.1,
                                maximum=1,
                                step=0.1,
                            )
                    with gr.Accordion(open=False, label="Image Processing"):
                        db_resolution = gr.Slider(
                            label="Max Resolution",
                            step=64,
                            minimum=128,
                            value=512,
                            maximum=2048,
                            elem_id="max_res",
                        )
                        db_hflip = gr.Checkbox(
                            label="Apply Horizontal Flip", value=False
                        )
                        db_dynamic_img_norm = gr.Checkbox(
                            label="Dynamic Image Normalization", value=False
                        )
                    with gr.Accordion(open=False, label="Prior Loss") as db_prior_loss_view:
                        db_prior_loss_scale = gr.Checkbox(
                            label="Scale Prior Loss", value=False
                        )
                        db_prior_loss_weight = gr.Slider(
                            label="Prior Loss Weight",
                            minimum=0.01,
                            maximum=1,
                            step=0.01,
                            value=0.75,
                        )
                        db_prior_loss_target = gr.Number(
                            label="Prior Loss Target",
                            value=100,
                            visible=False,
                        )
                        db_prior_loss_weight_min = gr.Slider(
                            label="Minimum Prior Loss Weight",
                            minimum=0.01,
                            maximum=1,
                            step=0.01,
                            value=0.1,
                            visible=False,
                        )
                    with gr.Accordion(open=False, label="Saving", elem_id="TabSave") as db_save_tab:
                        with gr.Column():
                            gr.HTML("General")
                            db_custom_model_name = gr.Textbox(
                                label="Custom Model Name",
                                value="",
                                placeholder="Enter a model name for saving checkpoints and lora models.",
                            )
                            db_save_safetensors = gr.Checkbox(
                                label="Save in .safetensors format",
                                value=True,
                                visible=False,
                            )
                            db_save_ema = gr.Checkbox(
                                label="Save EMA Weights to Generated Models", value=True
                            )
                            db_infer_ema = gr.Checkbox(
                                label="Use EMA Weights for Inference", value=False
                            )
                        with gr.Column():
                            gr.HTML("Checkpoints")
                            db_half_model = gr.Checkbox(label="Half Model", value=False)
                            db_use_subdir = gr.Checkbox(
                                label="Save Checkpoint to Subdirectory", value=True
                            )
                            db_save_ckpt_during = gr.Checkbox(
                                label="Generate a .ckpt file when saving during training."
                            )
                            db_save_ckpt_after = gr.Checkbox(
                                label="Generate a .ckpt file when training completes.",
                                value=True,
                            )
                            db_save_ckpt_cancel = gr.Checkbox(
                                label="Generate a .ckpt file when training is canceled."
                            )
                        with gr.Column(visible=False) as lora_save_col:
                            db_save_lora_during = gr.Checkbox(
                                label="Generate lora weights when saving during training."
                            )
                            db_save_lora_after = gr.Checkbox(
                                label="Generate lora weights when training completes.",
                                value=True,
                            )
                            db_save_lora_cancel = gr.Checkbox(
                                label="Generate lora weights when training is canceled."
                            )
                            db_save_lora_for_extra_net = gr.Checkbox(
                                label="Generate lora weights for extra networks."
                            )
                        with gr.Column():
                            gr.HTML("Diffusion Weights (training snapshots)")
                            db_save_state_during = gr.Checkbox(
                                label="Save separate diffusers snapshots when saving during training."
                            )
                            db_save_state_after = gr.Checkbox(
                                label="Save separate diffusers snapshots when training completes."
                            )
                            db_save_state_cancel = gr.Checkbox(
                                label="Save separate diffusers snapshots when training is canceled."
                            )
                    with gr.Accordion(open=False, label="Image Generation", elem_id="TabGenerate") as db_generate_tab:
                        gr.HTML(value="Class Generation Schedulers")
                        db_class_gen_method = gr.Dropdown(
                            label="Image Generation Library",
                            value="Native Diffusers",
                            choices=[
                                "A1111 txt2img (Euler a)",
                                "Native Diffusers",
                            ]
                        )
                        db_scheduler = gr.Dropdown(
                            label="Image Generation Scheduler",
                            value="DEISMultistep",
                            choices=get_scheduler_names(),
                        )
                        gr.HTML(value="Manual Class Generation")
                        with gr.Column():
                            db_generate_classes = gr.Button(value="Generate Class Images")
                            db_generate_graph = gr.Button(value="Generate Graph")
                            db_graph_smoothing = gr.Slider(
                                value=50,
                                label="Graph Smoothing Steps",
                                minimum=10,
                                maximum=500,
                            )
                            db_debug_buckets = gr.Button(value="Debug Buckets")
                            db_bucket_epochs = gr.Slider(
                                value=10,
                                step=1,
                                minimum=1,
                                maximum=1000,
                                label="Epochs to Simulate",
                            )
                            db_bucket_batch = gr.Slider(
                                value=1,
                                step=1,
                                minimum=1,
                                maximum=500,
                                label="Batch Size to Simulate",
                            )
                            db_generate_sample = gr.Button(value="Generate Sample Images")
                            db_sample_prompt = gr.Textbox(label="Sample Prompt")
                            db_sample_negative = gr.Textbox(label="Sample Negative Prompt")
                            db_sample_prompt_file = gr.Textbox(label="Sample Prompt File")
                            db_sample_width = gr.Slider(
                                label="Sample Width",
                                value=512,
                                step=64,
                                minimum=128,
                                maximum=2048,
                            )
                            db_sample_height = gr.Slider(
                                label="Sample Height",
                                value=512,
                                step=64,
                                minimum=128,
                                maximum=2048,
                            )
                            db_sample_seed = gr.Number(
                                label="Sample Seed", value=-1, precision=0
                            )
                            db_num_samples = gr.Slider(
                                label="Number of Samples to Generate",
                                value=1,
                                minimum=1,
                                maximum=1000,
                                step=1,
                            )
                            db_gen_sample_batch_size = gr.Slider(
                                label="Sample Batch Size",
                                value=1,
                                step=1,
                                minimum=1,
                                maximum=100,
                                interactive=True,
                            )
                            db_sample_steps = gr.Slider(
                                label="Sample Steps",
                                value=20,
                                minimum=1,
                                maximum=500,
                                step=1,
                            )
                            db_sample_scale = gr.Slider(
                                label="Sample CFG Scale",
                                value=7.5,
                                step=0.1,
                                minimum=1,
                                maximum=20,
                            )
                            with gr.Column(variant="panel", visible=has_face_swap()):
                                db_swap_faces = gr.Checkbox(label="Swap Sample Faces")
                                db_swap_prompt = gr.Textbox(label="Swap Prompt")
                                db_swap_negative = gr.Textbox(label="Swap Negative Prompt")
                                db_swap_steps = gr.Slider(label="Swap Steps", value=40)
                                db_swap_batch = gr.Slider(label="Swap Batch", value=40)

                            db_sample_txt2img = gr.Checkbox(
                                label="Use txt2img",
                                value=False,
                                visible=False  # db_sample_txt2img not implemented yet
                            )
                    with gr.Accordion(open=False, label="Extras"):
                        with gr.Column():
                            gr.HTML(value="Sanity Samples")
                            db_sanity_prompt = gr.Textbox(
                                label="Sanity Sample Prompt",
                                placeholder="A generic prompt used to generate a sample image "
                                            "to verify model fidelity.",
                            )
                            db_sanity_negative_prompt = gr.Textbox(
                                label="Sanity Sample Negative Prompt",
                                placeholder="A negative prompt for the generic sample image.",
                            )
                            db_sanity_seed = gr.Number(
                                label="Sanity Sample Seed", value=420420
                            )
                        with gr.Column() as db_misc_view:
                            gr.HTML(value="Miscellaneous")
                            db_pretrained_vae_name_or_path = gr.Textbox(
                                label="Pretrained VAE Name or Path",
                                placeholder="Leave blank to use base model VAE.",
                                value="",
                            )
                            db_use_concepts = gr.Checkbox(
                                label="Use Concepts List", value=False
                            )
                            db_concepts_path = gr.Textbox(
                                label="Concepts List",
                                placeholder="Path to JSON file with concepts to train.",
                            )
                            with gr.Row():
                                db_secret = gr.Textbox(
                                    label="API Key", value=get_secret, interactive=False
                                )
                                db_refresh_button = gr.Button(
                                    value=refresh_symbol, elem_id="refresh_secret"
                                )
                                db_clear_secret = gr.Button(
                                    value=delete_symbol, elem_id="clear_secret"
                                )
                        with gr.Column() as db_hook_view:
                            gr.HTML(value="Webhooks")
                            # In the future change this to something more generic and list the supported types
                            # from extensions.sd_dreambooth_extension.dreamboothWebhookTarget enum; for now, Discord is what I use ;)
                            # Add options to include notifications on training complete and exceptions that halt training
                            db_notification_webhook_url = gr.Textbox(
                                label="Discord Webhook",
                                placeholder="https://discord.com/api/webhooks/XXX/XXXX",
                                value="",
                            )
                            notification_webhook_test_btn = gr.Button(
                                value="Save and Test Webhook"
                            )
                        with gr.Column() as db_test_tab:
                            gr.HTML(value="Experimental Settings")
                            db_tomesd = gr.Slider(
                                value=0,
                                label="Token Merging (ToMe)",
                                minimum=0,
                                maximum=1,
                                step=0.1,
                            )
                            db_split_loss = gr.Checkbox(
                                label="Calculate Split Loss", value=True
                            )
                            db_disable_class_matching = gr.Checkbox(label="Disable Class Matching")
                            db_disable_logging = gr.Checkbox(label="Disable Logging")
                            db_deterministic = gr.Checkbox(label="Deterministic")
                            db_ema_predict = gr.Checkbox(label="Use EMA for prediction")
                            db_lora_use_buggy_requires_grad = gr.Checkbox(label="LoRA use buggy requires grad")
                            db_noise_scheduler = gr.Dropdown(
                                label="Noise scheduler",
                                value="DDPM",
                                choices=[
                                    "DDPM",
                                    "DEIS",
                                    "UniPC"
                                ]
                            )
                            db_update_extension = gr.Button(
                                value="Update Extension and Restart"
                            )

                            with gr.Column(variant="panel"):
                                gr.HTML(value="Bucket Cropping")
                                db_crop_src_path = gr.Textbox(label="Source Path")
                                db_crop_dst_path = gr.Textbox(label="Dest Path")
                                db_crop_max_res = gr.Slider(
                                    label="Max Res", value=512, step=64, maximum=2048
                                )
                                db_crop_bucket_step = gr.Slider(
                                    label="Bucket Steps", value=8, step=8, maximum=512
                                )
                                db_crop_dry = gr.Checkbox(label="Dry Run")
                                db_start_crop = gr.Button("Start Cropping")
            with gr.Column(variant="panel"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=110):
                        gr.HTML(value="<span class='hh'>Output</span>")
                    with gr.Column(scale=1, min_width=110):
                        db_check_progress_initial = gr.Button(
                            value=update_symbol,
                            elem_id="db_check_progress_initial",
                            visible=False,
                        )
                        # These two should be updated while doing things
                        db_active = gr.Checkbox(elem_id="db_active", value=False, visible=False)

                        ui_check_progress_initial = gr.Button(
                            value="Refresh", elem_id="ui_check_progress_initial", elem_classes="advBtn", size="sm"
                        )
                db_status = gr.HTML(elem_id="db_status", value="")
                db_progressbar = gr.HTML(elem_id="db_progressbar")
                db_gallery = gr.Gallery(
                    label="Output", show_label=False, elem_id="db_gallery", columns=4
                )
                db_preview = gr.Image(elem_id="db_preview", visible=False)
                db_prompt_list = gr.HTML(
                    elem_id="db_prompt_list", value="", visible=False
                )
                db_gallery_prompt = gr.HTML(elem_id="db_gallery_prompt", value="")
                db_check_progress = gr.Button(
                    "Check Progress", elem_id=f"db_check_progress", visible=False
                )
                db_update_params = gr.Button(
                    "Update Parameters", elem_id="db_update_params", visible=False
                )
                db_launch_error = gr.HTML(
                    elem_id="launch_errors", visible=False, value=get_launch_errors
                )

                def check_toggles(
                        use_lora, class_gen_method, lr_scheduler, train_unet, scale_prior
                ):
                    stop_text_encoder = update_stop_tenc(train_unet)
                    (
                        show_ema,
                        use_lora_extended,
                        lora_save,
                        lora_rank,
                        lora_lr,
                        standard_lr,
                        lora_model,
                        _,
                        _,
                        _
                    ) = disable_lora(use_lora)
                    (
                        lr_power,
                        lr_cycles,
                        lr_scale_pos,
                        lr_factor,
                        learning_rate_min,
                        lr_warmup_steps,
                    ) = lr_scheduler_changed(lr_scheduler)
                    scheduler = class_gen_method_changed(class_gen_method)
                    loss_min, loss_tgt = toggle_loss_items(scale_prior)
                    return (
                        stop_text_encoder,
                        show_ema,
                        use_lora_extended,
                        lora_save,
                        lora_rank,
                        lora_lr,
                        lora_model,
                        scheduler,
                        lr_power,
                        lr_cycles,
                        lr_scale_pos,
                        lr_factor,
                        learning_rate_min,
                        lr_warmup_steps,
                        loss_min,
                        loss_tgt,
                        standard_lr
                    )

                db_start_crop.click(
                    _js="db_start_crop",
                    fn=start_crop,
                    inputs=[
                        db_crop_src_path,
                        db_crop_dst_path,
                        db_crop_max_res,
                        db_crop_bucket_step,
                        db_crop_dry,
                    ],
                    outputs=[db_status, db_gallery],
                )

                db_update_params.click(
                    fn=check_toggles,
                    inputs=[
                        db_use_lora,
                        db_class_gen_method,
                        db_lr_scheduler,
                        db_train_unet,
                        db_prior_loss_scale,
                    ],
                    outputs=[
                        db_stop_text_encoder,
                        db_use_ema,
                        db_use_lora_extended,
                        lora_save_col,
                        lora_rank_col,
                        lora_lr_row,
                        lora_model_row,
                        db_scheduler,
                        db_lr_power,
                        db_lr_cycles,
                        db_lr_scale_pos,
                        db_lr_factor,
                        db_learning_rate_min,
                        db_lr_warmup_steps,
                        db_prior_loss_weight_min,
                        db_prior_loss_target,
                        standard_lr_row,
                    ],
                )

                db_update_extension.click(fn=update_extension, inputs=[], outputs=[])

                notification_webhook_test_btn.click(
                    fn=save_and_test_webhook,
                    inputs=[db_notification_webhook_url],
                    outputs=[db_status],
                )

                db_refresh_button.click(
                    fn=create_secret, inputs=[], outputs=[db_secret]
                )

                def update_stop_tenc(train_unet):
                    # If train unet enabled, read "hidden" value from stop_tenc and restore
                    if train_unet:
                        return gr.update(interactive=True)
                    else:
                        return gr.update(interactive=False)

                db_train_unet.change(
                    fn=update_stop_tenc,
                    inputs=[db_train_unet],
                    outputs=[db_stop_text_encoder],
                )

                def toggle_full_mixed_precision(full_mixed_precision):
                    if full_mixed_precision != "fp16":
                        return gr.update(visible=False)
                    else:
                        return gr.update(visible=True)

                db_mixed_precision.change(
                    fn=toggle_full_mixed_precision,
                    inputs=[db_mixed_precision],
                    outputs=[db_full_mixed_precision],
                )

                def update_model_options(model_type):
                    if model_type == "SDXL":
                        return gr.update(value=1024)
                    else:
                        return gr.update(value=512)

                db_model_type_select.change(
                    fn=update_model_options,
                    inputs=[db_model_type_select],
                    outputs=[db_resolution]
                )

                db_clear_secret.click(fn=clear_secret, inputs=[], outputs=[db_secret])

                # Elements to update when progress changes
                progress_elements = [
                    db_active,
                    db_progressbar,
                    db_preview,
                    db_gallery,
                    db_status,
                    db_prompt_list,
                    ui_check_progress_initial,
                ]

                db_check_progress.click(
                    fn=lambda: check_progress_call(),
                    show_progress=False,
                    inputs=[],
                    outputs=progress_elements
                )

                db_check_progress_initial.click(
                    fn=lambda: check_progress_call_initial(),
                    show_progress=False,
                    inputs=[],
                    outputs=progress_elements,
                )

                ui_check_progress_initial.click(
                    fn=lambda: check_progress_call(),
                    show_progress=False,
                    inputs=[],
                    outputs=progress_elements,
                )

            def format_updates():
                updates = check_updates()
                strings = []
                if updates is not None:
                    for key, value in updates.items():
                        rev = key
                        title = value[0]
                        author = value[1]
                        date = value[2]
                        url = value[3]
                        title = f"<div class='commitDiv'><h3>{title}</h3><span>{author} - {date} - <a href='{url}'>{rev}</a><br></div>"
                        strings.append(title)
                return "\n".join(strings)

            with gr.Row(variant="panel", elem_id="change_modal"):
                with gr.Row():
                    modal_title = gr.HTML("<h2>Changelog</h2>", elem_id="modal_title")
                    close_modal = gr.Button(value="X", elem_id="close_modal")
                with gr.Row():
                    modal_release_notes = gr.HTML(
                        "<h3><a href='https://github.com/d8ahazard/sd_dreambooth_extension/releases/latest'>Release notes</a></h3>",
                        elem_id="modal_notes",
                    )
                with gr.Column():
                    change_log = gr.HTML(format_updates(), elem_id="change_log")

        advanced_elements = [
            db_snapshot_row,
            db_create_from_hub,
            db_new_model_extract_ema,
            db_train_unfrozen,
            db_use_ema,
            db_freeze_clip_normalization,
            db_full_mixed_precision,
            db_offset_noise,
            db_weight_decay,
            db_tenc_weight_decay,
            db_tenc_grad_clip_norm,
            db_min_snr_gamma,
            db_use_dream,
            db_dream_detail_preservation,
            db_freeze_spectral_norm,
            db_pad_tokens,
            db_strict_tokens,
            db_max_token_length,
            db_epoch_pause_frequency,
            db_epoch_pause_time,
            db_batch_size_view,
            db_lr_scheduler,
            db_lr_warmup_steps,
            db_hflip,
            db_prior_loss_view,
            db_misc_view,
            db_hook_view,
            db_save_tab,
            db_generate_tab,
            db_test_tab,
            db_dynamic_img_norm,
            db_tomesd,
            db_split_loss,
            db_disable_class_matching,
            db_disable_logging,
            db_deterministic,
            db_ema_predict,
            db_lora_use_buggy_requires_grad,
            db_noise_scheduler,
            c1_class_guidance_scale,
            c1_class_infer_steps,
            c1_save_sample_negative_prompt,
            c1_sample_seed,
            c1_save_guidance_scale,
            c1_save_infer_steps,
            c2_class_guidance_scale,
            c2_class_infer_steps,
            c2_save_sample_negative_prompt,
            c2_sample_seed,
            c2_save_guidance_scale,
            c2_save_infer_steps,
            c3_class_guidance_scale,
            c3_class_infer_steps,
            c3_save_sample_negative_prompt,
            c3_sample_seed,
            c3_save_guidance_scale,
            c3_save_infer_steps,
            c4_class_guidance_scale,
            c4_class_infer_steps,
            c4_save_sample_negative_prompt,
            c4_sample_seed,
            c4_save_guidance_scale,
            c4_save_infer_steps,
        ]

        def toggle_advanced():
            global show_advanced
            show_advanced = False if show_advanced else True
            outputs = [gr.update(visible=True), gr.update(visible=False)]
            print(f"Advanced elements visible: {show_advanced}")
            for _ in advanced_elements:
                outputs.append(gr.update(visible=show_advanced))

            return outputs

        # Merge db_show advanced, db_hide_advanced, and advanced elements into one list
        db_show_advanced.click(
            fn=toggle_advanced,
            inputs=[],
            outputs=[db_hide_advanced, db_show_advanced, *advanced_elements]
        )

        db_hide_advanced.click(
            fn=toggle_advanced,
            inputs=[],
            outputs=[db_show_advanced, db_hide_advanced, *advanced_elements]
        )

        global preprocess_params

        # preprocess_params = [
        #     db_preprocess_path,
        #     db_preprocess_recursive
        # ]
        #
        # db_preprocess_path.change(
        #     fn=check_preprocess_path,
        #     inputs=[db_preprocess_path, db_preprocess_recursive],
        #     outputs=[db_status, db_gallery]
        # )

        db_gallery.select(load_image_caption, None, db_status)

        global params_to_save
        global params_to_load

        # List of all the things that we need to save
        # db_model_name must be first due to save_config() parsing
        params_to_save = [
            db_weight_decay,
            db_attention,
            db_cache_latents,
            db_clip_skip,
            db_concepts_path,
            db_custom_model_name,
            db_deterministic,
            db_disable_class_matching,
            db_disable_logging,
            db_ema_predict,
            db_tomesd,
            db_epoch_pause_frequency,
            db_epoch_pause_time,
            db_epochs,
            db_freeze_clip_normalization,
            db_full_mixed_precision,
            db_gradient_accumulation_steps,
            db_gradient_checkpointing,
            db_gradient_set_to_none,
            db_graph_smoothing,
            db_half_model,
            db_hflip,
            db_infer_ema,
            db_learning_rate,
            db_learning_rate_min,
            db_lora_learning_rate,
            db_lora_model_name,
            db_lora_txt_learning_rate,
            db_lora_txt_rank,
            db_lora_unet_rank,
            db_lora_use_buggy_requires_grad,
            db_lora_weight,
            db_lr_cycles,
            db_lr_factor,
            db_lr_power,
            db_lr_scale_pos,
            db_lr_scheduler,
            db_lr_warmup_steps,
            db_max_token_length,
            db_min_snr_gamma,
            db_use_dream,
            db_dream_detail_preservation,
            db_freeze_spectral_norm,
            db_mixed_precision,
            db_model_name,
            db_model_path,
            db_noise_scheduler,
            db_num_train_epochs,
            db_offset_noise,
            db_optimizer,
            db_pad_tokens,
            db_pretrained_vae_name_or_path,
            db_prior_loss_scale,
            db_prior_loss_target,
            db_prior_loss_weight,
            db_prior_loss_weight_min,
            db_resolution,
            db_revision,
            db_sample_batch_size,
            db_sanity_prompt,
            db_sanity_seed,
            db_save_ckpt_after,
            db_save_ckpt_cancel,
            db_save_ckpt_during,
            db_save_ema,
            db_save_embedding_every,
            db_save_lora_after,
            db_save_lora_cancel,
            db_save_lora_during,
            db_save_lora_for_extra_net,
            db_save_preview_every,
            db_save_safetensors,
            db_save_state_after,
            db_save_state_cancel,
            db_save_state_during,
            db_scheduler,
            db_shared_diffusers_path,
            db_shuffle_tags,
            db_snapshot,
            db_split_loss,
            db_src,
            db_stop_text_encoder,
            db_strict_tokens,
            db_dynamic_img_norm,
            db_tenc_grad_clip_norm,
            db_tenc_weight_decay,
            db_train_batch_size,
            db_train_imagic,
            db_train_unet,
            db_train_unfrozen,
            db_txt_learning_rate,
            db_use_concepts,
            db_use_ema,
            db_use_lora,
            db_use_lora_extended,
            db_use_shared_src,
            db_use_subdir,

            c1_class_data_dir,
            c1_class_guidance_scale,
            c1_class_infer_steps,
            c1_class_negative_prompt,
            c1_class_prompt,
            c1_class_token,
            c1_instance_data_dir,
            c1_instance_prompt,
            c1_instance_token,
            c1_n_save_sample,
            c1_num_class_images_per,
            c1_sample_seed,
            c1_save_guidance_scale,
            c1_save_infer_steps,
            c1_save_sample_negative_prompt,
            c1_save_sample_prompt,
            c1_save_sample_template,
            c2_class_data_dir,
            c2_class_guidance_scale,
            c2_class_infer_steps,
            c2_class_negative_prompt,
            c2_class_prompt,
            c2_class_token,
            c2_instance_data_dir,
            c2_instance_prompt,
            c2_instance_token,
            c2_n_save_sample,
            c2_num_class_images_per,
            c2_sample_seed,
            c2_save_guidance_scale,
            c2_save_infer_steps,
            c2_save_sample_negative_prompt,
            c2_save_sample_prompt,
            c2_save_sample_template,
            c3_class_data_dir,
            c3_class_guidance_scale,
            c3_class_infer_steps,
            c3_class_negative_prompt,
            c3_class_prompt,
            c3_class_token,
            c3_instance_data_dir,
            c3_instance_prompt,
            c3_instance_token,
            c3_n_save_sample,
            c3_num_class_images_per,
            c3_sample_seed,
            c3_save_guidance_scale,
            c3_save_infer_steps,
            c3_save_sample_negative_prompt,
            c3_save_sample_prompt,
            c3_save_sample_template,
            c4_class_data_dir,
            c4_class_guidance_scale,
            c4_class_infer_steps,
            c4_class_negative_prompt,
            c4_class_prompt,
            c4_class_token,
            c4_instance_data_dir,
            c4_instance_prompt,
            c4_instance_token,
            c4_n_save_sample,
            c4_num_class_images_per,
            c4_sample_seed,
            c4_save_guidance_scale,
            c4_save_infer_steps,
            c4_save_sample_negative_prompt,
            c4_save_sample_prompt,
            c4_save_sample_template,
        ]
        for element in params_to_save:
            setattr(element, "do_not_save_to_config", True)

        # Do not load these values when 'load settings' is clicked
        params_to_exclude = [
            db_model_name,
            db_epochs,
            db_model_path,
            db_revision,
            db_src,
            db_model_type,
            db_shared_diffusers_path,
        ]

        # Populate by the below method and handed out to other elements
        params_to_load = []
        save_keys = []
        ui_keys = []

        for param in params_to_save:
            var_name = [var_name for var_name, var in locals().items() if var is param]
            save_keys.append(var_name[0])
            if param not in params_to_exclude:
                ui_keys.append(var_name[0])
                params_to_load.append(param)

        ui_keys.append("db_status")
        params_to_load.append(db_status)
        from extensions.sd_dreambooth_extension.dreambooth.dataclasses import db_config
        db_config.save_keys = save_keys
        db_config.ui_keys = ui_keys

        db_save_params.click(
            _js="check_save", fn=save_config, inputs=params_to_save, outputs=[]
        )

        db_load_params.click(
            _js="db_start_load_params",
            fn=load_params,
            inputs=[db_model_name],
            outputs=params_to_load,
        )

        def toggle_new_rows(create_from):
            return gr.update(visible=create_from), gr.update(visible=not create_from)

        def toggle_loss_items(scale):
            return gr.update(visible=scale), gr.update(visible=scale)

        db_create_from_hub.change(
            fn=toggle_new_rows,
            inputs=[db_create_from_hub],
            outputs=[hub_row, local_row],
        )

        def toggle_shared_row(row):
            return gr.update(visible=row), gr.update(value="")

        db_use_shared_src.change(
            fn=toggle_shared_row,
            inputs=[db_use_shared_src],
            outputs=[shared_row, db_new_model_shared_src],
        )

        db_prior_loss_scale.change(
            fn=toggle_loss_items,
            inputs=[db_prior_loss_scale],
            outputs=[db_prior_loss_weight_min, db_prior_loss_target],
        )

        def disable_lora(x):
            use_ema = gr.update(interactive=not x)
            use_lora_extended = gr.update(visible=False)
            lora_save = gr.update(visible=x)
            lora_rank = gr.update(visible=x)
            lora_lr = gr.update(visible=x)
            standard_lr = gr.update(visible=not x)
            lora_model = gr.update(visible=x)
            if x:
                save_during = gr.update(label="Save LORA during training")
                save_after = gr.update(label="Save LORA after training")
                save_cancel = gr.update(label="Save LORA on cancel")
            else:
                save_during = gr.update(label="Save .safetensors during training")
                save_after = gr.update(label="Save .safetensors after training")
                save_cancel = gr.update(label="Save .safetensors on cancel")
            return (
                use_ema,
                use_lora_extended,
                lora_save,
                lora_rank,
                lora_lr,
                standard_lr,
                lora_model,
                save_during,
                save_after,
                save_cancel
            )

        def lr_scheduler_changed(sched):
            show_scale_pos = gr.update(visible=False)
            show_min_lr = gr.update(visible=False)
            show_lr_factor = gr.update(visible=False)
            show_lr_warmup = gr.update(visible=False)
            show_lr_power = gr.update(visible=sched == "polynomial")
            show_lr_cycles = gr.update(visible=sched == "cosine_with_restarts")
            scale_scheds = [
                "constant",
                "linear",
                "cosine_annealing",
                "cosine_annealing_with_restarts",
            ]
            if sched in scale_scheds:
                show_scale_pos = gr.update(visible=True)
            else:
                show_lr_warmup = gr.update(visible=True)
            if sched in ["cosine_annealing", "cosine_annealing_with_restarts"]:
                show_min_lr = gr.update(visible=True)
            if sched in ["linear", "constant"]:
                show_lr_factor = gr.update(visible=True)
            return (
                show_lr_power,
                show_lr_cycles,
                show_scale_pos,
                show_lr_factor,
                show_min_lr,
                show_lr_warmup,
            )

        def optimizer_changed(opti):
            show_adapt = "adapt" in opti
            adaptation_lr = gr.update(visible=show_adapt)
            return adaptation_lr

        def class_gen_method_changed(method):
            show_scheduler = method == "Native Diffusers"
            scheduler = gr.update(visible=show_scheduler)
            return scheduler

        db_use_lora.change(
            fn=disable_lora,
            inputs=[db_use_lora],
            outputs=[
                db_use_ema,
                db_use_lora_extended,
                lora_save_col,
                lora_rank_col,
                lora_lr_row,
                standard_lr_row,
                lora_model_row,
                db_save_ckpt_during,
                db_save_ckpt_after,
                db_save_ckpt_cancel
            ],
        )

        db_lr_scheduler.change(
            fn=lr_scheduler_changed,
            inputs=[db_lr_scheduler],
            outputs=[
                db_lr_power,
                db_lr_cycles,
                db_lr_scale_pos,
                db_lr_factor,
                db_learning_rate_min,
                db_lr_warmup_steps,
            ],
        )

        db_class_gen_method.change(
            fn=class_gen_method_changed,
            inputs=[db_class_gen_method],
            outputs=[db_scheduler],
        )

        db_model_name.change(
            _js="clear_loaded",
            fn=load_model_params,
            inputs=[db_model_name],
            outputs=[
                db_model_info,
                db_model_path,
                db_revision,
                db_epochs,
                db_model_type,
                db_has_ema,
                db_src,
                db_shared_diffusers_path,
                db_snapshot,
                db_lora_model_name,
                db_status,
            ],
        )

        db_use_concepts.change(
            fn=lambda x: {concept_tab: gr_show(x is True)},
            inputs=[db_use_concepts],
            outputs=[concept_tab],
        )

        db_generate_graph.click(
            _js="db_start_logs",
            fn=log_parser.parse_logs,
            inputs=[db_model_name, gr.Checkbox(value=True, visible=False)],
            outputs=[db_gallery, db_prompt_list],
        )

        db_debug_buckets.click(
            _js="db_start_buckets",
            fn=debug_buckets,
            inputs=[db_model_name, db_bucket_epochs, db_bucket_batch],
            outputs=[db_status, db_status],
        )

        db_performance_wizard.click(
            fn=performance_wizard,
            _js="db_start_pwizard",
            inputs=[db_model_name],
            outputs=[
                db_attention,
                db_gradient_checkpointing,
                db_gradient_accumulation_steps,
                db_mixed_precision,
                db_cache_latents,
                db_optimizer,
                db_sample_batch_size,
                db_train_batch_size,
                db_stop_text_encoder,
                db_use_lora,
                db_use_ema,
                db_save_preview_every,
                db_save_embedding_every,
                db_status,
            ],
        )

        db_generate_sample.click(
            fn=wrap_gpu_call(generate_samples),
            _js="db_start_sample",
            inputs=[
                db_model_name,
                db_sample_prompt,
                db_sample_prompt_file,
                db_sample_negative,
                db_sample_width,
                db_sample_height,
                db_num_samples,
                db_sample_batch_size,
                db_sample_seed,
                db_sample_steps,
                db_sample_scale,
                db_sample_txt2img,
                db_scheduler,
                db_swap_faces,
                db_swap_prompt,
                db_swap_negative,
                db_swap_steps,
                db_swap_batch,
            ],
            outputs=[db_gallery, db_prompt_list, db_status],
        )

        db_generate_checkpoint.click(
            _js="db_start_checkpoint",
            fn=wrap_gpu_call(ui_gen_ckpt),
            inputs=[db_model_name],
            outputs=[db_status],
        )

        def set_gen_ckpt():
            status.do_save_model = True

        def set_gen_sample():
            status.do_save_samples = True

        db_generate_checkpoint_during.click(fn=set_gen_ckpt, inputs=[], outputs=[])

        db_train_sample.click(fn=set_gen_sample, inputs=[], outputs=[])

        db_create_model.click(
            fn=wrap_gpu_call(create_model),
            _js="db_start_create",
            inputs=[
                db_new_model_name,
                db_new_model_src,
                db_new_model_shared_src,
                db_create_from_hub,
                db_new_model_url,
                db_new_model_token,
                db_new_model_extract_ema,
                db_train_unfrozen,
                db_model_type_select
            ],
            outputs=[
                db_model_name,
                db_model_path,
                db_revision,
                db_epochs,
                db_src,
                db_shared_diffusers_path,
                db_has_ema,
                db_model_type,
                db_resolution,
                db_status,
            ],
        )

        db_train_model.click(
            fn=wrap_gpu_call(start_training),
            _js="db_start_train",
            inputs=[db_model_name, db_class_gen_method],
            outputs=[db_lora_model_name, db_revision, db_epochs, db_gallery, db_status],
        )

        db_generate_classes.click(
            _js="db_start_classes",
            fn=wrap_gpu_call(ui_classifiers),
            inputs=[db_model_name, db_class_gen_method],
            outputs=[db_gallery, db_status],
        )

        db_cancel.click(
            fn=lambda: status.interrupt(),
            inputs=[],
            outputs=[],
        )

    return ((dreambooth_interface, "Dreambooth", "dreambooth_v2"),)


def build_concept_panel(concept: int):
    with gr.Tab(label="Instance Images"):
        instance_data_dir = gr.Textbox(
            label="Directory",
            placeholder="Path to directory with input images",
            elem_id=f"idd{concept}",
        )
        instance_prompt = gr.Textbox(label="Prompt", value="[filewords]")
        gr.HTML(
            value="Use [filewords] here to read prompts from caption files/filename, or a prompt to describe your training images.<br>"
                  "If using [filewords], your instance and class tokens will be inserted into the prompt as necessary for training.",
            elem_classes="hintHtml")
        instance_token = gr.Textbox(label="Instance Token")
        gr.HTML(
            value="If using [filewords] above, this is the unique word used for your subject, like 'fydodog' or 'ohwx'.",
            elem_classes="hintHtml")
        class_token = gr.Textbox(label="Class Token")
        gr.HTML(
            value="If using [filewords] above, this is the generic word used for your subject, like 'dog' or 'person'.",
            elem_classes="hintHtml")

    with gr.Tab(label="Class Images"):
        class_data_dir = gr.Textbox(
            label="Directory",
            placeholder="(Optional) Path to directory with "
                        "classification/regularization images",
            elem_id=f"cdd{concept}",
        )
        class_prompt = gr.Textbox(label="Prompt", value="[filewords]")
        gr.HTML(
            value="Use [filewords] here to read prompts from caption files/filename, or a prompt to describe your training images.<br>"
                  "If using [filewords], your class token will be inserted into the file prompts if it is not found.",
            elem_classes="hintHtml")

        class_negative_prompt = gr.Textbox(
            label="Negative Prompt"
        )
        num_class_images_per = gr.Slider(
            label="Class Images Per Instance Image", value=0, precision=0
        )
        gr.HTML(
            value="For every instance image, this many classification images will be used/generated. Leave at 0 to disable.",
            elem_classes="hintHtml")
        class_guidance_scale = gr.Slider(
            label="Classification CFG Scale", value=7.5, maximum=12, minimum=1, step=0.1
        )
        class_infer_steps = gr.Slider(
            label="Classification Steps", value=40, minimum=10, maximum=200, step=1
        )
    with gr.Tab(label="Sample Images"):
        save_sample_prompt = gr.Textbox(
            label="Sample Image Prompt",
            value='[filewords]'
        )
        gr.HTML(
            value="A prompt to generate samples from, or use [filewords] here to randomly select prompts from the existing instance prompt(s).<br>"
                  "If using [filewords], your instance token will be inserted into the file prompts if it is not found.",
            elem_classes="hintHtml")

        save_sample_negative_prompt = gr.Textbox(
            label="Sample Negative Prompt"
        )
        sample_template = gr.Textbox(
            label="Sample Prompt Template File",
            placeholder="Enter the path to a txt file containing sample prompts.",
        )
        gr.HTML(value="When enabled the above prompt and negative prompt will be ignored.",
                elem_classes="hintHtml")
        n_save_sample = gr.Slider(
            label="Number of Samples to Generate", value=1, maximum=100, step=1
        )
        sample_seed = gr.Number(label="Sample Seed", value=-1, precision=0)
        save_guidance_scale = gr.Slider(
            label="Sample CFG Scale", value=7.5, maximum=12, minimum=1, step=0.1
        )
        save_infer_steps = gr.Slider(
            label="Sample Steps", value=20, minimum=10, maximum=200, step=1
        )
    return [
        instance_data_dir,
        class_data_dir,
        instance_prompt,
        class_prompt,
        save_sample_prompt,
        sample_template,
        instance_token,
        class_token,
        num_class_images_per,
        class_negative_prompt,
        class_guidance_scale,
        class_infer_steps,
        save_sample_negative_prompt,
        n_save_sample,
        sample_seed,
        save_guidance_scale,
        save_infer_steps,
    ]


script_callbacks.on_ui_tabs(on_ui_tabs)
