import base64
import io
import time
import uvicorn
from fastapi import FastAPI
from dreambooth import conversion, dreambooth
import gradio as gr
from webui import wrap_gradio_gpu_call
import modules.script_callbacks as script_callbacks
from pydantic import BaseModel
from typing import Optional
class dreamboothParameters(BaseModel):
    db_pretrained_model_name_or_path: str
    db_pretrained_vae_name_or_path: Optional[str] = ""
    db_instance_data_dir: str
    db_class_data_dir: Optional[str] = ""
    db_instance_prompt: Optional[str] = ""
    db_use_filename_as_label: Optional[bool] = False
    db_use_txt_as_label: Optional[bool] = False
    db_class_prompt: Optional[str] = ""
    db_save_sample_prompt: Optional[str]  = ""
    db_save_sample_negative_prompt: Optional[str]  = ""
    db_n_save_sample: Optional[int] = 500
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
    db_use_8bit_adam: Optional[bool] = False
    db_adam_beta1: Optional[float] = 0.9
    db_adam_beta2: Optional[float] = 0.999
    db_adam_weight_decay: Optional[float] = 0.01
    db_adam_epsilon: Optional[float] = 0.00000001
    db_max_grad_norm: Optional[int] = 1
    db_save_preview_every: Optional[int] = 500
    db_save_embedding_every: Optional[int] =500
    db_mixed_precision: Optional[str] = "no"
    db_not_cache_latents: Optional[bool] = True
    db_concepts_list: Optional[str] = ""
    db_use_cpu: Optional[bool] = False
    db_pad_tokens: Optional[bool] = True
    db_hflip: Optional[bool] = True


def dreamBoothAPI(demo: gr.Blocks, app: FastAPI):
    @app.post("/dreambooth/createModel")
    async def createModel(name: str, source: str, scheduler: str):
        print("Creating new Checkpoint: " + name)
        fn = conversion.extract_checkpoint(name, source, scheduler)
    @app.post("/dreambooth/start_straining")
    async def start_training(paras: dreamboothParameters):
        print("Starting Training")
        fn = wrap_gradio_gpu_call(dreambooth.start_training(
            paras.db_pretrained_model_name_or_path,
            paras.db_pretrained_vae_name_or_path,
            paras.db_instance_data_dir,
            paras.db_class_data_dir,
            paras.db_instance_prompt,
            paras.db_use_filename_as_label,
            paras.db_use_txt_as_label,
            paras.db_class_prompt,
            paras.db_save_sample_prompt,
            paras.db_save_sample_negative_prompt,
            paras.db_n_save_sample,
            paras.db_save_guidance_scale,
            paras.db_save_infer_steps,
            paras.db_num_class_images,
            paras.db_resolution,
            paras.db_center_crop,
            paras.db_train_text_encoder,
            paras.db_train_batch_size,
            paras.db_sample_batch_size,
            paras.db_num_train_epochs,
            paras.db_max_train_steps,
            paras.db_gradient_accumulation_steps,
            paras.db_gradient_checkpointing,
            paras.db_learning_rate,
            paras.db_scale_lr,
            paras.db_lr_scheduler,
            paras.db_lr_warmup_steps,
            paras.db_use_8bit_adam,
            paras.db_adam_beta1,
            paras.db_adam_beta2,
            paras.db_adam_weight_decay,
            paras.db_adam_epsilon,
            paras.db_max_grad_norm,
            paras.db_save_preview_every,
            paras.db_save_embedding_every,
            paras.db_mixed_precision,
            paras.db_not_cache_latents,
            paras.db_concepts_list,
            paras.db_use_cpu,
            paras.db_pad_tokens,
            paras.db_hflip))


script_callbacks.on_app_started(dreamBoothAPI)

print("Dreambooth API layer loaded")