import asyncio
from typing import Optional

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

import modules.script_callbacks as script_callbacks
from extensions.sd_dreambooth_extension.dreambooth import dreambooth
from extensions.sd_dreambooth_extension.dreambooth.sd_to_diff import extract_checkpoint
from webui import wrap_gradio_gpu_call


class DreamboothParameters(BaseModel):
    db_half_model: bool = False,
    db_use_concepts: bool = False,
    db_pretrained_model_name_or_path: str
    db_pretrained_vae_name_or_path: Optional[str] = ""
    db_instance_data_dir: str
    db_class_data_dir: Optional[str] = ""
    db_instance_prompt: Optional[str] = ""
    db_class_prompt: Optional[str] = ""
    db_instance_token: str = ""
    db_class_token: str = ""
    db_save_sample_prompt: Optional[str] = ""
    db_save_sample_negative_prompt: Optional[str] = ""
    db_n_save_sample: Optional[int] = 500
    db_sample_seed: Optional[int] = -1
    db_save_guidance_scale: Optional[float] = 7.5
    db_save_infer_steps: Optional[int] = 40
    db_num_class_images: Optional[int] = 0
    db_resolution: Optional[int] = 512
    db_center_crop: Optional[bool] = False
    db_train_text_encoder: Optional[bool] = True
    db_train_batch_size: Optional[int] = 1
    db_sample_batch_size: Optional[int] = 1
    db_num_train_epochs: Optional[int] = 1
    db_max_train_steps: Optional[int] = 1000
    db_gradient_accumulation_steps: Optional[int] = 1
    db_gradient_checkpointing: Optional[bool] = True
    db_learning_rate: Optional[float] = 0.000005
    db_scale_lr: Optional[bool] = False
    db_lr_scheduler: Optional[str] = "constant"
    db_lr_warmup_steps: Optional[int] = 0
    db_attention: Optional[str] = "default"
    db_use_8bit_adam: Optional[bool] = False
    db_adam_beta1: Optional[float] = 0.9
    db_adam_beta2: Optional[float] = 0.999
    db_adam_weight_decay: Optional[float] = 0.01
    db_adam_epsilon: Optional[float] = 0.00000001
    db_save_preview_every: Optional[int] = 500
    db_save_embedding_every: Optional[int] = 500
    db_mixed_precision: Optional[str] = "no"
    db_not_cache_latents: Optional[bool] = True
    db_concepts_list: Optional[str] = ""
    db_use_cpu: Optional[bool] = False
    db_pad_tokens: Optional[bool] = True
    db_max_token_length: Optional[int] = 75
    db_hflip: Optional[bool] = True
    db_use_ema: Optional[bool] = False
    db_class_negative_prompt: Optional[str] = ""
    db_class_guidance_scale: Optional[float] = 7.5
    db_class_infer_steps: Optional[int] = 60
    db_shuffle_after_epoch: Optional[bool] = False


def dreamBoothAPI(demo: gr.Blocks, app: FastAPI):
    @app.post("/dreambooth/createModel")
    async def createModel(
                name,
                source,
                scheduler,
                model_url,
                hub_token):
        print("Creating new Checkpoint: " + name)
        fn = extract_checkpoint(name, source, scheduler, model_url, hub_token)

    @app.post("/dreambooth/start_straining")
    async def start_training(params: DreamboothParameters):
        print("Starting Training")
        task = asyncio.create_task(train_model(params))
        return {"status": "finished"}

    # TODO: Add a new method that saves a config from all the params, or accepts an existing config file
    async def train_model(params: DreamboothParameters):
        fn = wrap_gradio_gpu_call(dreambooth.start_training(
            params.db_pretrained_model_name_or_path,
            params.db_half_model
        ))

    # TODO: Add methods to compile a checkpoint, generate sample image


script_callbacks.on_app_started(dreamBoothAPI)

print("Dreambooth API layer loaded")
