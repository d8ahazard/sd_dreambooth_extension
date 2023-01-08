import time
from typing import List

import gradio as gr

from extensions.sd_dreambooth_extension.dreambooth.db_config import save_config, from_file
from extensions.sd_dreambooth_extension.dreambooth.db_shared import status
from extensions.sd_dreambooth_extension.dreambooth.db_webhook import save_and_test_webhook
from extensions.sd_dreambooth_extension.dreambooth.diff_to_sd import compile_checkpoint
from extensions.sd_dreambooth_extension.dreambooth.finetune_utils import generate_prompts
from extensions.sd_dreambooth_extension.dreambooth.sd_to_diff import extract_checkpoint
from extensions.sd_dreambooth_extension.dreambooth.secret import get_secret, create_secret, clear_secret
from extensions.sd_dreambooth_extension.dreambooth.utils import get_db_models, list_attention, \
    list_floats, get_lora_models, wrap_gpu_call, parse_logs
from extensions.sd_dreambooth_extension.scripts import dreambooth
from extensions.sd_dreambooth_extension.scripts.dreambooth import performance_wizard, \
    training_wizard, training_wizard_person, load_model_params, ui_classifiers, ui_samples, debug_buckets
from modules import script_callbacks, sd_models
from modules.ui import gr_show, create_refresh_button

params_to_save = []
refresh_symbol = '\U0001f504'  # 🔄
delete_symbol = '\U0001F5D1'  # 🗑️
update_symbol = '\U0001F51D'  # 🠝


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
        time_since_start = time.time() - status.time_start
        eta = (time_since_start / progress)
        eta_relative = eta - time_since_start
        if (eta_relative > threshold and progress > 0.02) or force_display:
            if eta_relative > 86400:
                days = eta_relative // 86400
                remainder = days * 86400
                eta_relative -= remainder
                return f"{label}{days}:{time.strftime('%H:%M:%S', time.gmtime(eta_relative))}"
            if eta_relative > 3600:
                return label + time.strftime('%H:%M:%S', time.gmtime(eta_relative))
            elif eta_relative > 60:
                return label + time.strftime('%M:%S', time.gmtime(eta_relative))
            else:
                return label + time.strftime('%Ss', time.gmtime(eta_relative))
        else:
            return ""


def check_progress_call():
    """
    Check the progress from share dreamstate and return appropriate UI elements.
    @return:
    pspan: Progress bar span contents
    preview: Preview Image/Visibility
    gallery: Gallery Image/Visibility
    textinfo_result: Primary status
    sample_prompts: List = A list of prompts corresponding with gallery contents
    check_progress_initial: Hides the manual 'check progress' button
    """
    if status.job_count == 0:
        return "", gr.update(visible=False, value=None), gr.update(visible=True), gr_show(True), gr_show(True), \
               gr_show(False)
    progress = 0

    if status.job_count > 0:
        progress += status.job_no / status.job_count

    time_left = calc_time_left(progress, 1, " ETA: ", status.time_left_force_display)
    if time_left != "":
        status.time_left_force_display = True

    progress = min(progress, 1)

    progressbar = f"""<div class='progressDiv'><div class='progress' style="overflow:visible;width:{progress * 100}%;white-space:nowrap;">{"&nbsp;" * 2 + str(int(progress * 100)) + "%" + time_left if progress > 0.01 else ""}</div></div>"""
    status.set_current_image()
    image = status.current_image

    if image is None:
        preview = gr.update(visible=False, value=None)
        gallery = gr.update(visible=True)
    else:
        if isinstance(image, List):
            if len(image) > 1:
                status.current_image = None
                preview = gr.update(visible=False, value=None)
                gallery = gr.update(visible=True, value=image)
            else:
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
    return pspan, preview, gallery, textinfo_result, gr.update(value=prompts), gr_show(False)


def check_progress_call_initial():
    status.job_count = -1
    status.current_latent = None
    status.current_image = None
    status.textinfo = None
    status.textinfo2 = None
    status.time_start = time.time()
    status.time_left_force_display = False
    pspan, preview, gallery, textinfo_result, prompts_result, pbutton_result = check_progress_call()
    return pspan, gr_show(False), gr.update(value=[]), textinfo_result, gr.update(value=[]), gr_show(False)


def ui_gen_ckpt(model_name: str):
    if isinstance(model_name, List):
        model_name = model_name[0]
    if model_name == "" or model_name is None:
        return "Please select a model."
    config = from_file(model_name)
    half = config.half_model
    use_subdir = config.use_subdir
    lora_path = config.lora_model_name
    lora_alpha = config.lora_weight
    lora_txt_alpha = config.lora_txt_weight
    custom_model_name = config.custom_model_name
    res = compile_checkpoint(model_name, half, use_subdir, lora_path, lora_alpha, lora_txt_alpha, custom_model_name,
                             True, True)
    return res


def on_ui_tabs():
    with gr.Blocks() as dreambooth_interface:
        with gr.Row(equal_height=True):
            db_load_params = gr.Button(value='Load Settings', elem_id="db_load_params")
            db_save_params = gr.Button(value="Save Settings", elem_id="db_save_config")
            db_train_model = gr.Button(value="Train", variant='primary', elem_id="db_train")
            db_generate_checkpoint = gr.Button(value="Generate Ckpt", elem_id="db_gen_ckpt")
            db_generate_checkpoint_during = gr.Button(value="Save Weights", elem_id="db_gen_ckpt_during")
            db_train_sample = gr.Button(value="Generate Samples", elem_id="db_train_sample")
            db_cancel = gr.Button(value="Cancel", elem_id="db_cancel")
        with gr.Row().style(equal_height=False):
            with gr.Column(variant="panel"):
                gr.HTML(value="<span class='hh'>Model Selection</span>")
                with gr.Row():
                    db_model_name = gr.Dropdown(label='Model', choices=sorted(get_db_models()))
                    create_refresh_button(db_model_name, get_db_models, lambda: {
                        "choices": sorted(get_db_models())},
                                          "refresh_db_models")
                with gr.Row(visible=False) as lora_model_row:
                    db_lora_model_name = gr.Dropdown(label='Lora Model', choices=sorted(get_lora_models()))
                    create_refresh_button(db_lora_model_name, get_lora_models, lambda: {
                        "choices": sorted(get_lora_models())},
                                          "refresh_lora_models")

                with gr.Row():
                    gr.HTML(value="Loaded Model:")
                    db_model_path = gr.HTML()
                with gr.Row():
                    gr.HTML(value="Model Revision:")
                    db_revision = gr.HTML(elem_id="db_revision")
                with gr.Row():
                    gr.HTML(value="Model Epoch:")
                    db_epochs = gr.HTML(elem_id="db_epochs")
                with gr.Row():
                    gr.HTML(value="V2 Model:")
                    db_v2 = gr.HTML(elem_id="db_v2")
                with gr.Row():
                    gr.HTML(value="Has EMA:")
                    db_has_ema = gr.HTML(elem_id="db_has_ema")
                with gr.Row():
                    gr.HTML(value="Source Checkpoint:")
                    db_src = gr.HTML()
                with gr.Row():
                    gr.HTML(value="Scheduler:")
                    db_scheduler = gr.HTML()


            with gr.Column(variant="panel"):
                gr.HTML(value="<span class='hh'>Input</span>")
                with gr.Tab("Create"):
                    with gr.Column():
                        db_create_model = gr.Button(value="Create Model", variant='primary')
                    db_new_model_name = gr.Textbox(label="Name")
                    db_create_from_hub = gr.Checkbox(label="Import Model from Huggingface Hub", value=False)
                    with gr.Column(visible=False) as hub_row:
                        db_new_model_url = gr.Textbox(label="Model Path", placeholder="runwayml/stable-diffusion-v1-5")
                        db_new_model_token = gr.Textbox(label="HuggingFace Token", value="")
                    with gr.Column(visible=True) as local_row:
                        with gr.Row():
                            db_new_model_src = gr.Dropdown(label='Source Checkpoint',
                                                           choices=sorted(get_sd_models()))
                            create_refresh_button(db_new_model_src, get_sd_models, lambda: {
                                "choices": sorted(get_sd_models())},
                                                  "refresh_sd_models")
                    db_new_model_extract_ema = gr.Checkbox(label='Extract EMA Weights', value=False)
                    db_new_model_scheduler = gr.Dropdown(label='Scheduler', choices=["pndm", "lms", "euler",
                                                                                     "euler-ancestral", "dpm", "ddim"],
                                                         value="ddim")
                with gr.Tab("Settings"):
                    db_performance_wizard = gr.Button(value="Performance Wizard (WIP)")
                    with gr.Accordion(open=True, label="Basic"):
                        with gr.Column():
                            gr.HTML(value="General")
                            db_use_lora = gr.Checkbox(label="Use LORA", value=False)
                            db_train_imagic_only = gr.Checkbox(label="Train Imagic Only", value=False)
                            db_train_inpainting = gr.Checkbox(label="Train Inpainting Model", value=False,
                                                              visible=False)
                            db_use_txt2img = gr.Checkbox(label="Generate Classification Images Using txt2img",
                                                         value=True)
                        with gr.Column():
                            gr.HTML(value="Intervals")
                            db_num_train_epochs = gr.Number(label="Training Steps Per Image (Epochs)", precision=0,
                                                            value=100)
                            db_epoch_pause_frequency = gr.Number(label='Pause After N Epochs', value=0)
                            db_epoch_pause_time = gr.Number(label='Amount of time to pause between Epochs (s)',
                                                            value=60)
                            db_save_embedding_every = gr.Number(
                                label='Save Model Frequency (Epochs)', value=25,
                                precision=0)
                            db_save_preview_every = gr.Number(
                                label='Save Preview(s) Frequency (Epochs)', value=5,
                                precision=0)

                        with gr.Column():
                            gr.HTML(value="Batching")
                            db_train_batch_size = gr.Number(label="Batch Size", precision=0, value=1)
                            db_gradient_accumulation_steps = gr.Number(label="Gradient Accumulation Steps",
                                                                       precision=0,
                                                                       value=1)
                            db_sample_batch_size = gr.Number(label="Class Batch Size", precision=0, value=1)
                            db_gradient_set_to_none = gr.Checkbox(label="Set Gradients to None When Zeroing",
                                                                  value=True)
                            db_gradient_checkpointing = gr.Checkbox(label="Gradient Checkpointing", value=True)

                        schedulers = ["linear", "linear_with_warmup", "cosine", "cosine_annealing",
                                      "cosine_annealing_with_restarts", "cosine_with_restarts", "polynomial",
                                      "constant", "constant_with_warmup"]
                        with gr.Column():
                            gr.HTML(value="Learning Rate")
                            db_lr_scheduler = gr.Dropdown(label="Learning Rate Scheduler", value="constant_with_warmup",
                                                          choices=schedulers)

                            db_learning_rate = gr.Number(label='Learning Rate', value=2e-6)
                            db_learning_rate_min = gr.Number(label='Min Learning Rate', value=1e-6, visible=False)
                            db_lr_cycles = gr.Number(label="Number of Hard Resets", value=1, precision=0, visible=False)
                            db_lr_factor = gr.Number(label="Constant/Linear Starting Factor", value=0.5, precision=2, visible=False)
                            db_lr_power = gr.Number(label="Polynomial Power", value=1.0, precision=1, visible=False)
                            db_lr_scale_pos = gr.Slider(label="Scale Position", value=0.5, minimum=0, maximum=1, step=0.05, visible=False)
                            with gr.Row(visible=False) as lora_lr_row:
                                db_lora_learning_rate = gr.Number(label='Lora UNET Learning Rate', value=2e-4)
                                db_lora_txt_learning_rate = gr.Number(label='Lora Text Encoder Learning Rate',
                                                                      value=2e-4)
                            db_lr_warmup_steps = gr.Number(label="Learning Rate Warmup Steps", precision=0, value=0)


                        with gr.Column():
                            gr.HTML(value="Image Processing")
                            db_resolution = gr.Number(label="Resolution", precision=0, value=512)
                            db_center_crop = gr.Checkbox(label="Center Crop", value=False)
                            db_hflip = gr.Checkbox(label="Apply Horizontal Flip", value=False)
                            db_sanity_prompt = gr.Textbox(label="Sanity Sample Prompt", placeholder="A generic prompt "
                                                                                                    "used to generate"
                                                                                                    " a sample image "
                                                                                                    "to verify model "
                                                                                                    "fidelity.")
                            db_sanity_seed = gr.Number(label="Sanity Sample Seed", value=420420)
                        with gr.Column():
                            gr.HTML(value="Miscellaneous")
                            db_pretrained_vae_name_or_path = gr.Textbox(label='Pretrained VAE Name or Path',
                                                                        placeholder="Leave blank to use base model VAE.",
                                                                        value="")

                            db_use_concepts = gr.Checkbox(label="Use Concepts List", value=False)
                            db_concepts_path = gr.Textbox(label="Concepts List",
                                                          placeholder="Path to JSON file with concepts to train.")
                            with gr.Row():
                                db_secret = gr.Textbox(label="API Key", value=get_secret, interactive=False)
                                db_refresh_button = gr.Button(value=refresh_symbol, elem_id="refresh_secret")
                                db_clear_secret = gr.Button(value=delete_symbol, elem_id="clear_secret")
                            with gr.Column():
                                # In the future change this to something more generic and list the supported types
                                # from DreamboothWebhookTarget enum; for now, Discord is what I use ;)
                                # Add options to include notifications on training complete and exceptions that halt training
                                db_notification_webhook_url = gr.Textbox(label="Discord Webhook",
                                                                      placeholder="https://discord.com/api/webhooks/XXX/XXXX",
                                                                      value="")
                                notification_webhook_test_btn = gr.Button(value="Save and Test Webhook")


                    with gr.Accordion(open=False, label="Advanced"):
                        with gr.Row():
                            with gr.Column():
                                with gr.Column():
                                    gr.HTML(value="Tuning")
                                    db_use_ema = gr.Checkbox(label="Use EMA", value=False)
                                    db_use_8bit_adam = gr.Checkbox(label="Use 8bit Adam", value=False)
                                    db_mixed_precision = gr.Dropdown(label="Mixed Precision", value="no",
                                                                     choices=list_floats())
                                    db_attention = gr.Dropdown(
                                        label="Memory Attention", value="default",
                                        choices=list_attention())
                                    db_cache_latents = gr.Checkbox(label="Cache Latents", value=True)
                                    db_stop_text_encoder = gr.Slider(label="Step Ratio of Text Encoder Training", minimum=0, maximum=1, step=0.01, value=1, visible=True)
                                    db_clip_skip = gr.Slider(label="Clip Skip", value=1, minimum=1, maximum=12, step=1)
                                    db_prior_loss_weight = gr.Number(label="Prior Loss Weight", value=1.0, precision=1)
                                    db_pad_tokens = gr.Checkbox(label="Pad Tokens", value=True)
                                    db_shuffle_tags = gr.Checkbox(label="Shuffle Tags", value=True)
                                    db_max_token_length = gr.Slider(label="Max Token Length", minimum=75, maximum=300,
                                                                    step=75)

                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.HTML(value="")
                with gr.Tab("Concepts") as concept_tab:
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            db_train_wizard_person = gr.Button(value="Training Wizard (Person)")
                            db_train_wizard_object = gr.Button(value="Training Wizard (Object/Style)")
                        with gr.Tab("Concept 1"):
                            c1_instance_data_dir, c1_class_data_dir, c1_instance_prompt, \
                            c1_class_prompt, c1_num_class_images, c1_save_sample_prompt, c1_save_sample_template, c1_instance_token, \
                            c1_class_token, c1_num_class_images_per, c1_class_negative_prompt, c1_class_guidance_scale, \
                            c1_class_infer_steps, c1_save_sample_negative_prompt, c1_n_save_sample, c1_sample_seed, \
                            c1_save_guidance_scale, c1_save_infer_steps = build_concept_panel()

                        with gr.Tab("Concept 2"):
                            c2_instance_data_dir, c2_class_data_dir, c2_instance_prompt, \
                            c2_class_prompt, c2_num_class_images, c2_save_sample_prompt, c2_save_sample_template, c2_instance_token, \
                            c2_class_token, c2_num_class_images_per, c2_class_negative_prompt, c2_class_guidance_scale, \
                            c2_class_infer_steps, c2_save_sample_negative_prompt, c2_n_save_sample, c2_sample_seed, \
                            c2_save_guidance_scale, c2_save_infer_steps = build_concept_panel()

                        with gr.Tab("Concept 3"):
                            c3_instance_data_dir, c3_class_data_dir, c3_instance_prompt, \
                            c3_class_prompt, c3_num_class_images, c3_save_sample_prompt, c3_save_sample_template, c3_instance_token, \
                            c3_class_token, c3_num_class_images_per, c3_class_negative_prompt, c3_class_guidance_scale, \
                            c3_class_infer_steps, c3_save_sample_negative_prompt, c3_n_save_sample, c3_sample_seed, \
                            c3_save_guidance_scale, c3_save_infer_steps = build_concept_panel()
                with gr.Tab("Saving"):
                    with gr.Column():
                        gr.HTML("General")
                        db_custom_model_name = gr.Textbox(label="Custom Model Name", value="",
                                                          placeholder="Enter a model name for saving checkpoints and lora models.")
                    with gr.Column():
                        gr.HTML("Checkpoints")
                        db_half_model = gr.Checkbox(label="Half Model", value=False)
                        db_use_subdir = gr.Checkbox(label="Save Checkpoint to Subdirectory", value=True)
                        db_save_ckpt_during = gr.Checkbox(label="Generate a .ckpt file when saving during training.")
                        db_save_ckpt_after = gr.Checkbox(label="Generate a .ckpt file when training completes.", value=True)
                        db_save_ckpt_cancel = gr.Checkbox(label="Generate a .ckpt file when training is canceled.")
                    with gr.Column(visible=False) as lora_save_col:
                        gr.HTML("Lora")
                        db_lora_weight = gr.Slider(label="Lora Weight", value=1, minimum=0.1, maximum=1, step=0.1)
                        db_lora_txt_weight = gr.Slider(label="Lora Text Weight", value=1, minimum=0.1, maximum=1,
                                                       step=0.1)
                        db_save_lora_during = gr.Checkbox(label="Generate lora weights when saving during training.")
                        db_save_lora_after = gr.Checkbox(label="Generate lora weights when training completes.", value=True)
                        db_save_lora_cancel = gr.Checkbox(label="Generate lora weights when training is canceled.")
                    with gr.Column():
                        gr.HTML("Diffusion Weights")
                        db_save_state_during = gr.Checkbox(
                            label="Save separate diffusers snapshots when saving during training.")
                        db_save_state_after = gr.Checkbox(
                            label="Save separate diffusers snapshots when training completes.")
                        db_save_state_cancel = gr.Checkbox(
                            label="Save separate diffusers snapshots when training is canceled.")
                with gr.Tab("Generate"):
                    with gr.Column():
                        db_generate_classes = gr.Button(value="Generate Class Images")
                        db_generate_prompts = gr.Button(value="Preview Prompts")
                        db_generate_graph = gr.Button(value="Generate Graph")
                        db_graph_smoothing = gr.Number(value=50, label="Graph Smoothing Steps")
                        db_debug_buckets = gr.Button(value="Debug Buckets")
                        db_generate_sample = gr.Button(value="Generate Sample Images")
                        db_sample_prompt = gr.Textbox(label="Sample Prompt")
                        db_sample_negative = gr.Textbox(label="Sample Negative Prompt")
                        db_sample_seed = gr.Number(label="Sample Seed", value=-1, precision=0)
                        db_num_samples = gr.Number(label="Number of Samples to Generate", value=1, precision=0)
                        db_sample_steps = gr.Number(label="Sample Steps", value=60, precision=0)
                        db_sample_scale = gr.Number(label="Sample CFG Scale", value=7.5, precision=2)

            with gr.Column(variant="panel"):
                gr.HTML(value="<span class='hh'>Output</span>")
                ui_check_progress_initial = gr.Button(value=update_symbol, elem_id="ui_check_progress_initial")
                db_check_progress_initial = gr.Button(value=update_symbol, elem_id="db_check_progress_initial", visible=False)
                # These two should be updated while doing things
                db_status = gr.HTML(elem_id="db_status", value="")
                db_progressbar = gr.HTML(elem_id="db_progressbar")
                db_gallery = gr.Gallery(label='Output', show_label=False, elem_id='db_gallery').style(grid=4)
                db_preview = gr.Image(elem_id='db_preview', visible=False)
                db_prompt_list = gr.HTML(elem_id="db_prompt_list", value="", visible=False)
                db_gallery_prompt = gr.HTML(elem_id="db_gallery_prompt", value="")
                db_check_progress = gr.Button("Check Progress", elem_id=f"db_check_progress", visible=False)
                db_update_params = gr.Button("Update Parameters", elem_id="db_update_params", visible=False)

                def check_toggles(use_ema, use_lora, lr_scheduler, stop_text_encoder):
                    pad_tokens = update_pad_tokens(stop_text_encoder)
                    show_ema, lora_save, lora_lr, lora_model = disable_ema(use_lora)
                    if not use_lora and use_ema:
                        disable_lora(use_ema)
                    lr_power, lr_cycles, lr_scale_pos, lr_factor, learning_rate_min, lr_warmup_steps = toggle_lr_min(
                        lr_scheduler)
                    return pad_tokens,\
                        show_ema,\
                        lora_save,\
                        lora_lr,\
                        lora_model,\
                        lr_power,\
                        lr_cycles,\
                        lr_scale_pos,\
                        lr_factor,\
                        learning_rate_min,\
                        lr_warmup_steps

                db_update_params.click(
                    fn=check_toggles,
                    inputs=[db_use_ema, db_use_lora, db_lr_scheduler, db_stop_text_encoder],
                    outputs=[db_pad_tokens,
                             db_use_ema,
                             lora_save_col,
                             lora_lr_row,
                             lora_model_row,
                             db_lr_power,
                             db_lr_cycles,
                             db_lr_scale_pos,
                             db_lr_factor,
                             db_learning_rate_min,
                             db_lr_warmup_steps]

                )

                notification_webhook_test_btn.click(
                    fn=save_and_test_webhook,
                    inputs=[db_notification_webhook_url],
                    outputs=[db_status]
                )

                db_refresh_button.click(
                    fn=create_secret,
                    inputs=[],
                    outputs=[db_secret]
                )


                def update_pad_tokens(x):
                    if x == 0:
                        return gr.update(visible=True)
                    else:
                        return gr.update(value=True)


                db_stop_text_encoder.change(
                    fn=update_pad_tokens,
                    inputs=[db_stop_text_encoder],
                    outputs=[db_pad_tokens]
                )

                db_clear_secret.click(
                    fn=clear_secret,
                    inputs=[],
                    outputs=[db_secret]
                )

                db_check_progress.click(
                    fn=lambda: check_progress_call(),
                    show_progress=False,
                    inputs=[],
                    outputs=[db_progressbar, db_preview, db_gallery, db_status, db_prompt_list, ui_check_progress_initial],
                )

                db_check_progress_initial.click(
                    fn=lambda: check_progress_call_initial(),
                    show_progress=False,
                    inputs=[],
                    outputs=[db_progressbar, db_preview, db_gallery, db_status, db_prompt_list, ui_check_progress_initial],
                )

                ui_check_progress_initial.click(
                    fn=lambda: check_progress_call(),
                    show_progress=False,
                    inputs=[],
                    outputs=[db_progressbar, db_preview, db_gallery, db_status, db_prompt_list,
                             ui_check_progress_initial],
                )

        global params_to_save

        params_to_save = [
            db_model_name,
            db_attention,
            db_cache_latents,
            db_center_crop,
            db_clip_skip,
            db_concepts_path,
            db_custom_model_name,
            db_epochs,
            db_epoch_pause_frequency,
            db_epoch_pause_time,
            db_gradient_accumulation_steps,
            db_gradient_checkpointing,
            db_gradient_set_to_none,
            db_graph_smoothing,
            db_half_model,
            db_has_ema,
            db_hflip,
            db_learning_rate,
            db_learning_rate_min,
            db_lora_learning_rate,
            db_lora_model_name,
            db_lora_txt_learning_rate,
            db_lora_txt_weight,
            db_lora_weight,
            db_lr_cycles,
            db_lr_factor,
            db_lr_power,
            db_lr_scale_pos,
            db_lr_scheduler,
            db_lr_warmup_steps,
            db_max_token_length,
            db_mixed_precision,
            db_model_path,
            db_num_train_epochs,
            db_pad_tokens,
            db_pretrained_vae_name_or_path,
            db_prior_loss_weight,
            db_resolution,
            db_revision,
            db_sample_batch_size,
            db_sanity_prompt,
            db_sanity_seed,
            db_save_ckpt_after,
            db_save_ckpt_cancel,
            db_save_ckpt_during,
            db_save_embedding_every,
            db_save_lora_after,
            db_save_lora_cancel,
            db_save_lora_during,
            db_save_preview_every,
            db_save_state_after,
            db_save_state_cancel,
            db_save_state_during,
            db_scheduler,
            db_src,
            db_shuffle_tags,
            db_train_batch_size,
            db_train_imagic_only,
            db_stop_text_encoder,
            db_use_8bit_adam,
            db_use_concepts,
            db_use_ema,
            db_use_lora,
            db_use_subdir,
            db_v2,
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
            c1_num_class_images,
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
            c2_num_class_images,
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
            c3_num_class_images,
            c3_num_class_images_per,
            c3_sample_seed,
            c3_save_guidance_scale,
            c3_save_infer_steps,
            c3_save_sample_negative_prompt,
            c3_save_sample_prompt,
            c3_save_sample_template
        ]

        db_save_params.click(
            _js="check_save",
            fn=save_config,
            inputs=params_to_save,
            outputs=[]
        )



        db_load_params.click(
            _js="db_start_load_params",
            fn=dreambooth.load_params,
            inputs=[
                db_model_name
            ],
            outputs=[
                db_attention,
                db_cache_latents,
                db_center_crop,
                db_clip_skip,
                db_concepts_path,
                db_custom_model_name,
                db_epoch_pause_frequency,
                db_epoch_pause_time,
                db_gradient_accumulation_steps,
                db_gradient_checkpointing,
                db_gradient_set_to_none,
                db_graph_smoothing,
                db_half_model,
                db_hflip,
                db_learning_rate,
                db_learning_rate_min,
                db_lora_learning_rate,
                db_lora_model_name,
                db_lora_txt_learning_rate,
                db_lora_txt_weight,
                db_lora_weight,
                db_lr_cycles,
                db_lr_factor,
                db_lr_power,
                db_lr_scale_pos,
                db_lr_scheduler,
                db_lr_warmup_steps,
                db_max_token_length,
                db_mixed_precision,
                db_num_train_epochs,
                db_pad_tokens,
                db_pretrained_vae_name_or_path,
                db_prior_loss_weight,
                db_resolution,
                db_sample_batch_size,
                db_sanity_prompt,
                db_sanity_seed,
                db_save_ckpt_after,
                db_save_ckpt_cancel,
                db_save_ckpt_during,
                db_save_embedding_every,
                db_save_lora_after,
                db_save_lora_cancel,
                db_save_lora_during,
                db_save_preview_every,
                db_save_state_after,
                db_save_state_cancel,
                db_save_state_during,
                db_shuffle_tags,
                db_train_batch_size,
                db_train_imagic_only,
                db_stop_text_encoder,
                db_use_8bit_adam,
                db_use_concepts,
                db_use_ema,
                db_use_lora,
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
                c1_num_class_images,
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
                c2_num_class_images,
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
                c3_num_class_images,
                c3_num_class_images_per,
                c3_sample_seed,
                c3_save_guidance_scale,
                c3_save_infer_steps,
                c3_save_sample_negative_prompt,
                c3_save_sample_prompt,
                c3_save_sample_template,
                db_status
            ]
        )

        def toggle_new_rows(create_from):
            return gr.update(visible=create_from), gr.update(visible=not create_from)

        db_create_from_hub.change(
            fn=toggle_new_rows,
            inputs=[db_create_from_hub],
            outputs=[hub_row, local_row],
        )

        def disable_ema(x):
            return gr.update(interactive=not x), gr.update(visible=x), gr.update(visible=x), gr.update(visible=x)

        def disable_lora(x):
            db_use_lora.interactive = not x

        def toggle_lr_min(sched):
            show_scale_pos = gr.update(visible=False)
            show_min_lr = gr.update(visible=False)
            show_lr_factor = gr.update(visible=False)
            show_lr_warmup = gr.update(visible=False)
            show_lr_power = gr.update(visible=sched == "polynomial")
            show_lr_cycles = gr.update(visible=sched == "cosine_with_restarts")
            scale_scheds = ["constant", "linear", "cosine_annealing", "cosine_annealing_with_restarts"]
            if sched in scale_scheds:
                show_scale_pos = gr.update(visible=True)
            else:
                show_lr_warmup = gr.update(visible=True)
            if sched == "cosine_annealing" or sched == "cosine_annealing_with_restarts":
                show_min_lr = gr.update(visible=True)
            if sched == "linear" or sched == "constant":
                show_lr_factor = gr.update(visible=True)
            return show_lr_power, show_lr_cycles, show_scale_pos, show_lr_factor, show_min_lr, show_lr_warmup

        db_use_lora.change(
            fn=disable_ema,
            inputs=[db_use_lora],
            outputs=[db_use_ema, lora_save_col, lora_lr_row, lora_model_row],
        )

        db_lr_scheduler.change(
            fn=toggle_lr_min,
            inputs=[db_lr_scheduler],
            outputs=[db_lr_power, db_lr_cycles, db_lr_scale_pos, db_lr_factor, db_learning_rate_min, db_lr_warmup_steps]
        )

        db_use_ema.change(
            fn=disable_lora,
            inputs=[db_use_ema],
            outputs=[db_use_lora],
        )

        db_model_name.change(
            _js="clear_loaded",
            fn=load_model_params,
            inputs=[db_model_name],
            outputs=[db_model_path, db_revision, db_epochs, db_v2, db_has_ema, db_src, db_scheduler, db_status]
        )

        db_use_concepts.change(
            fn=lambda x: {
                concept_tab: gr_show(x is True)
            },
            inputs=[db_use_concepts],
            outputs=[
                concept_tab
            ]
        )

        db_generate_prompts.click(
            _js="db_start_prompts",
            fn=generate_prompts,
            inputs=[db_model_name],
            outputs=[db_status]
        )

        db_generate_graph.click(
            fn=parse_logs,
            inputs=[db_model_name, gr.Checkbox(value=True, visible=False)],
            outputs=[db_gallery, db_prompt_list]
        )

        db_debug_buckets.click(
            _js="db_start_buckets",
            fn=debug_buckets,
            inputs=[db_model_name],
            outputs=[db_gallery, db_prompt_list, db_status]
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
                db_sample_batch_size,
                db_train_batch_size,
                db_stop_text_encoder,
                db_use_8bit_adam,
                db_use_lora,
                db_use_ema,
                db_status
            ]
        )

        db_train_wizard_person.click(
            fn=training_wizard_person,
            _js="db_start_twizard",
            inputs=[
                db_model_name
            ],
            outputs=[
                db_num_train_epochs,
                c1_num_class_images_per,
                c2_num_class_images_per,
                c3_num_class_images_per,
                db_status
            ]
        )

        db_train_wizard_object.click(
            fn=training_wizard,
            _js="db_start_twizard",
            inputs=[
                db_model_name
            ],
            outputs=[
                db_num_train_epochs,
                c1_num_class_images_per,
                c2_num_class_images_per,
                c3_num_class_images_per,
                db_status
            ]
        )

        db_generate_sample.click(
            fn=wrap_gpu_call(ui_samples),
            _js="db_start_sample",
            inputs=[db_model_name,
                    db_sample_prompt,
                    db_num_samples,
                    db_sample_batch_size,
                    db_lora_model_name,
                    db_lora_weight,
                    db_lora_txt_weight,
                    db_sample_negative,
                    db_sample_seed,
                    db_sample_steps,
                    db_sample_scale,
                    db_resolution],
            outputs=[db_gallery, db_prompt_list, db_status]
        )

        db_generate_checkpoint.click(
            _js="db_start_checkpoint",
            fn=wrap_gpu_call(ui_gen_ckpt),
            inputs=[
                db_model_name
            ],
            outputs=[
                db_status
            ]
        )

        def set_gen_ckpt():
            status.do_save_model = True

        def set_gen_sample():
            status.do_save_samples = True

        db_generate_checkpoint_during.click(
            fn=set_gen_ckpt,
            inputs=[],
            outputs=[]
        )

        db_train_sample.click(
            fn=set_gen_sample,
            inputs=[],
            outputs=[]
        )

        db_create_model.click(
            fn=wrap_gpu_call(extract_checkpoint),
            _js="db_start_create",
            inputs=[
                db_new_model_name,
                db_new_model_src,
                db_new_model_scheduler,
                db_create_from_hub,
                db_new_model_url,
                db_new_model_token,
                db_new_model_extract_ema
            ],
            outputs=[
                db_model_name, db_model_path, db_revision, db_epochs, db_scheduler, db_src, db_has_ema, db_v2, db_resolution,
                db_status
            ]
        )

        db_train_model.click(
            fn=wrap_gpu_call(dreambooth.start_training),
            _js="db_start_train",
            inputs=[
                db_model_name,
                db_use_txt2img
            ],
            outputs=[
                db_lora_model_name,
                db_revision,
                db_epochs,
                db_gallery,
                db_status
            ]
        )

        db_generate_classes.click(
            _js="db_start_classes",
            fn=wrap_gpu_call(ui_classifiers),
            inputs=[db_model_name, db_use_txt2img],
            outputs=[db_gallery, db_status]
        )

        db_cancel.click(
            fn=lambda: status.interrupt(),
            inputs=[],
            outputs=[],
        )

    return (dreambooth_interface, "Dreambooth", "dreambooth_interface"),


def build_concept_panel():
    with gr.Column():
        gr.HTML(value="Directories")
        instance_data_dir = gr.Textbox(label='Dataset Directory',
                                       placeholder="Path to directory with input images")
        class_data_dir = gr.Textbox(label='Classification Dataset Directory',
                                    placeholder="(Optional) Path to directory with "
                                                "classification/regularization images")
    with gr.Column():
        gr.HTML(value="Filewords")
        instance_token = gr.Textbox(label='Instance Token',
                                    placeholder="When using [filewords], this is the subject to use when building prompts.")
        class_token = gr.Textbox(label='Class Token',
                                 placeholder="When using [filewords], this is the class to use when building prompts.")

    with gr.Column():
        gr.HTML(value="Prompts")
        instance_prompt = gr.Textbox(label="Instance Prompt",
                                     placeholder="Optionally use [filewords] to read image "
                                                 "captions from files.")
        class_prompt = gr.Textbox(label="Class Prompt",
                                  placeholder="Optionally use [filewords] to read image "
                                              "captions from files.")
        save_sample_prompt = gr.Textbox(label="Sample Image Prompt",
                                        placeholder="Leave blank to use instance prompt. "
                                                    "Optionally use [filewords] to base "
                                                    "sample captions on instance images.")
        class_negative_prompt = gr.Textbox(label="Classification Image Negative Prompt")
        sample_template = gr.Textbox(label="Sample Prompt Template File",
                                     placeholder="Enter the path to a txt file containing sample prompts.")
        save_sample_negative_prompt = gr.Textbox(label="Sample Negative Prompt")

    with gr.Column():
        gr.HTML("Image Generation")
        num_class_images = gr.Number(label='Total Number of Class/Reg Images', value=0, precision=0, visible=False)
        num_class_images_per = gr.Number(label='Class Images Per Instance Image', value=0, precision=0)
        class_guidance_scale = gr.Number(label="Classification CFG Scale", value=7.5, max=12, min=1, precision=2)
        class_infer_steps = gr.Number(label="Classification Steps", value=40, min=10, max=200, precision=0)
        n_save_sample = gr.Number(label="Number of Samples to Generate", value=1, precision=0)
        sample_seed = gr.Number(label="Sample Seed", value=-1, precision=0)
        save_guidance_scale = gr.Number(label="Sample CFG Scale", value=7.5, max=12, min=1, precision=2)
        save_infer_steps = gr.Number(label="Sample Steps", value=40, min=10, max=200, precision=0)
    return [instance_data_dir, class_data_dir, instance_prompt, class_prompt,
            num_class_images,
            save_sample_prompt, sample_template, instance_token, class_token, num_class_images_per, class_negative_prompt,
            class_guidance_scale, class_infer_steps, save_sample_negative_prompt, n_save_sample, sample_seed,
            save_guidance_scale, save_infer_steps]


script_callbacks.on_ui_tabs(on_ui_tabs)
