import asyncio
import json
import traceback
from typing import Optional, List

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

import modules.script_callbacks as script_callbacks
from extensions.sd_dreambooth_extension.dreambooth import dream_state
from extensions.sd_dreambooth_extension.dreambooth.db_concept import Concept
from extensions.sd_dreambooth_extension.dreambooth.db_config import from_file, save_json, DreamboothConfig
from extensions.sd_dreambooth_extension.dreambooth.diff_to_sd import compile_checkpoint
from extensions.sd_dreambooth_extension.dreambooth.sd_to_diff import extract_checkpoint
from extensions.sd_dreambooth_extension.scripts import dreambooth
from extensions.sd_dreambooth_extension.scripts.dreambooth import generate_sample_img
from extensions.sd_dreambooth_extension.dreambooth.utils import wrap_gpu_call


class DreamboothParameters(BaseModel):
    model_name: str = "",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    adam_weight_decay: float = 0.01,
    attention: str = "default",
    center_crop: bool = True,
    concepts_path: str = "",
    custom_model_name: str = "",
    epoch_pause_frequency: int = 0,
    epoch_pause_time: int = 0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    half_model: bool = False,
    has_ema: bool = False,
    hflip: bool = False,
    learning_rate: float = 0.00000172,
    lora_learning_rate: float = 1e-4,
    lora_txt_learning_rate: float = 5e-5,
    lora_txt_weight: float = 1.0,
    lora_weight: float = 1.0,
    lr_scheduler: str = 'constant',
    lr_warmup_steps: int = 0,
    max_token_length: int = 75,
    max_train_steps: int = 1000,
    mixed_precision: str = "fp16",
    model_path: str = "",
    not_cache_latents = False,
    num_train_epochs: int = 1,
    pad_tokens: bool = True,
    pretrained_vae_name_or_path: str = "",
    prior_loss_weight: float = 1.0,
    resolution: int = 512,
    revision: int = 0,
    sample_batch_size: int = 1,
    save_ckpt_after: bool = True,
    save_ckpt_cancel: bool = True,
    save_ckpt_during: bool = True,
    save_class_txt: bool = False,
    save_embedding_every: int = 500,
    save_lora_after: bool = True,
    save_lora_cancel: bool = True,
    save_lora_during: bool = True,
    save_preview_every: int = 500,
    save_state_after: bool = False,
    save_state_cancel: bool = False,
    save_state_during: bool = False,
    save_use_global_counts: bool = False,
    save_use_epochs: bool = False,
    scale_lr: bool = False,
    scheduler: str = "ddim",
    src: str = "",
    shuffle_tags: bool = False,
    train_batch_size: int = 1,
    train_text_encoder: bool = True,
    use_8bit_adam: bool = True,
    use_concepts: bool = False,
    use_cpu: bool = False,
    use_ema: bool = True,
    use_lora: bool = False,
    v2: bool = False,
    c1_class_data_dir: str = "",
    c1_class_guidance_scale: float = 7.5,
    c1_class_infer_steps: int = 60,
    c1_class_negative_prompt: str = "",
    c1_class_prompt: str = "",
    c1_class_token: str = "",
    c1_instance_data_dir: str = "",
    c1_instance_prompt: str = "",
    c1_instance_token: str = "",
    c1_max_steps: int = -1,
    c1_n_save_sample: int = 1,
    c1_num_class_images: int = 0,
    c1_sample_seed: int = -1,
    c1_save_guidance_scale: float = 7.5,
    c1_save_infer_steps: int = 60,
    c1_save_sample_negative_prompt: str = "",
    c1_save_sample_prompt: str = "",
    c1_save_sample_template: str = "",
    c2_class_data_dir: str = "",
    c2_class_guidance_scale: float = 7.5,
    c2_class_infer_steps: int = 60,
    c2_class_negative_prompt: str = "",
    c2_class_prompt: str = "",
    c2_class_token: str = "",
    c2_instance_data_dir: str = "",
    c2_instance_prompt: str = "",
    c2_instance_token: str = "",
    c2_max_steps: int = -1,
    c2_n_save_sample: int = 1,
    c2_num_class_images: int = 0,
    c2_sample_seed: int = -1,
    c2_save_guidance_scale: float = 7.5,
    c2_save_infer_steps: int = 60,
    c2_save_sample_negative_prompt: str = "",
    c2_save_sample_prompt: str = "",
    c2_save_sample_template: str = "",
    c3_class_data_dir: str = "",
    c3_class_guidance_scale: float = 7.5,
    c3_class_infer_steps: int = 60,
    c3_class_negative_prompt: str = "",
    c3_class_prompt: str = "",
    c3_class_token: str = "",
    c3_instance_data_dir: str = "",
    c3_instance_prompt: str = "",
    c3_instance_token: str = "",
    c3_max_steps: int = -1,
    c3_n_save_sample: int = 1,
    c3_num_class_images: int = 0,
    c3_sample_seed: int = -1,
    c3_save_guidance_scale: float = 7.5,
    c3_save_infer_steps: int = 60,
    c3_save_sample_negative_prompt: str = "",
    c3_save_sample_prompt: str = "",
    c3_save_sample_template: str = "",
    concepts_list = [Concept]


def dreamBoothAPI(demo: gr.Blocks, app: FastAPI):
    @app.post("/dreambooth/createModel")
    async def createModel(
            db_new_model_name,
            db_new_model_src,
            db_new_model_scheduler,
            db_create_from_hub,
            db_new_model_url,
            db_new_model_token,
            db_new_model_extract_ema):
        print("Creating new Checkpoint: " + db_new_model_name)
        fn = extract_checkpoint(db_new_model_name,
                                db_new_model_src,
                                db_new_model_scheduler,
                                db_create_from_hub,
                                db_new_model_url,
                                db_new_model_token,
                                db_new_model_extract_ema)

    @app.post("/dreambooth/start_training")
    async def start_training(params: DreamboothParameters = None, model_name: str = "", lora_model_name: str = "",
                             lora_weight: int = 1, lora_txt_weight: int = 1, use_imagic: bool = False,
                             use_subdir: bool = False, custom_name: str = "", use_tx2img: bool = False):
        print("Starting Training")
        if params is not None:
            model_name = params.model_name
            lora_weight = params.lora_weight
            lora_txt_weight = params.lora_txt_weight

        task = asyncio.create_task(train_model(model_name, lora_model_name, lora_weight, lora_txt_weight, use_imagic,
                                               use_subdir, custom_name, use_tx2img))
        return {"status": "finished"}

    async def train_model(db_model_name,
                          db_lora_model_name,
                          db_lora_weight,
                          db_lora_txt_weight,
                          db_train_imagic_only,
                          db_use_subdir,
                          db_custom_model_name,
                          db_use_txt2img):

        wrap_gpu_call(dreambooth.start_training(
            db_model_name,
            db_lora_model_name,
            db_lora_weight,
            db_lora_txt_weight,
            db_train_imagic_only,
            db_use_subdir,
            db_custom_model_name,
            db_use_txt2img
        ))

    @app.get("/dreambooth/status")
    async def check_status():
        return {"current_state": f"{json.dumps(dream_state.status.dict())}"}

    @app.get("/dreambooth/model_config")
    async def get_model_config(model_name):
        cfg = from_file(model_name)
        return {"model_config": f"{cfg.__dict__}"}

    @app.post("/dreambooth/model_config")
    async def set_model_config(model_name: str, model_cfg: DreamboothParameters):
        try:
            print("Create config")
            config = DreamboothConfig(*model_cfg)
            print("Save")
            config.save()
            print("Saved?")
            return {"model_config": f"{json.dumps(config.__dict__)}"}
        except Exception as e:
            traceback.print_exc()
            return {"Exception saving model": f"{e}"}

    @app.get("/dreambooth/get_checkpoint")
    async def get_checkpoint(model_name: str, lora_model_name: str, lora_weight: int = 1, lora_text_weight: int = 1):
        config = from_file(model_name)
        ckpt_result = compile_checkpoint(model_name, config.half_model, False,lora_model_name,lora_weight,
                                         lora_text_weight, "", False, True)
        path = ""
        if "Checkpoint compiled successfully" in ckpt_result:
            path = ckpt_result.split(":")[1]
        return {"checkpoint_path": f"{path}"}

    @app.post("/dreambooth/get_images")
    async def generate_image(model_name: str, sample_prompt: str, num_images: int = 1, batch_size: int = 1,
                             lora_model_path: str = "", lora_weight: float = 1.0,lora_txt_weight: float = 1.0,
                             negative_prompt: str = "", steps: int = 60, scale: float = 7.5):
        images = generate_sample_img(model_name, sample_prompt,negative_prompt, -1,num_images, steps, scale)



script_callbacks.on_app_started(dreamBoothAPI)

print("Dreambooth API layer loaded")
