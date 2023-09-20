from pyexpat import model
from anyio import get_all_backends
from httpx import get
from numpy import save
import torch
import torch.backends.cuda
import torch.backends.cudnn
from dreambooth import shared
from dreambooth.dataclasses.db_config import DreamboothConfig
from dreambooth.shared import DreamState
from dreambooth.shared import db_model_config
from dreambooth.utils.model_utils import (
    disable_safe_unpickle,
    enable_safe_unpickle,
    xformerify,
    torch2ify,
)
from dreambooth.utils.text_utils import encode_hidden_state
from dreambooth.utils.utils import (cleanup, printm,)
from dreambooth.webhook import send_training_update
import accelerate
import torch
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from diffusers.models.attention_processor import LoRAAttnProcessor2_0, LoRAAttnProcessor
from diffusers.utils import logging as dl, randn_tensor
from torch.cuda.profiler import profile
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os

class TrainUtils:
    def __init__(self, pipeline, accelerator):
        self.pipeline = pipeline
        self.accelerator = accelerator
        self.model_config = DreamboothConfig()
        self.model_type = self.model_config.model_type
        self.model_path = self.model_config.model_path
        self.precision = self.model_config.mixed_precision
        self.save_lora =self.model_config.save_lora_during
        self.save_checkpoint = self.model_config.save_ckpt_during
        self.save_difusers = self.model_config.save_state_during
        model_dtypes = self.get_model_dtypes()
        self.tenc_dtype = model_dtypes["tenc_dtype"]
        self.weight_dtype = model_dtypes["weight_dtype"]
        self.vae_dtype = model_dtypes["vae_dtype"]

    def get_model_dtypes(self, precision=None, model_type=None):
        precision = self.precision if precision is None else precision
        model_type = self.model_type if model_type is None else model_type

        weight_dtype = torch.float32
        tenc_dtype = torch.float32
        vae_dtype = torch.float32

        if precision == "bf16":
            weight_dtype = torch.bfloat16
        elif precision == "fp16":
            weight_dtype = torch.float16

        if model_type == "SDXL":
            vae_dtype = torch.float32

        model_dtypes = {
            "weight_dtype": weight_dtype,
            "tenc_dtype": tenc_dtype,
            "vae_dtype": vae_dtype
        }

        self.weight_dtype = weight_dtype
        self.tenc_dtype = tenc_dtype
        self.vae_dtype = vae_dtype

        return model_dtypes

    def prepare_pipeline_for_inference(self, pipeline=None):
        accelerator = self.accelerator
        pipeline = self.pipeline if pipeline is None else pipeline
        model_type = self.model_type
        weight_dtype = self.weight_dtype
        tenc_dtype = self.tenc_dtype
        vae_dtype = self.vae_dtype
        
        # Send all the models to the same device with correct dtypes
        pipeline.unet.to(accelerator.device, weight_dtype)
        pipeline.text_encoder.to(accelerator.device, tenc_dtype)
        pipeline.vae.to(accelerator.device, vae_dtype)
        if model_type == "SDXL":
            pipeline.text_encoder_two.to(accelerator.device, tenc_dtype)

        # Get the models ready for inference
        pipeline.unet.eval()
        pipeline.text_encoder.eval()
        if model_type == "SDXL":
            pipeline.text_encoder_two.eval()
        pipeline.vae.eval()

        return pipeline
    
    def prepare_pipeline_for_training(self, pipeline=None):
        accelerator = self.accelerator
        pipeline = self.pipeline if pipeline is None else pipeline
        model_type = self.model_type
        weight_dtype = self.weight_dtype
        tenc_dtype = self.tenc_dtype
        vae_dtype = self.vae_dtype

        # Send all the models to the same device with correct dtypes
        # They should already be there but might as well be safe
        pipeline.unet.to(accelerator.device, dtype=weight_dtype)
        pipeline.text_encoder.to(accelerator.device, dtype=tenc_dtype)
        pipeline.vae.to(accelerator.device, dtype=vae_dtype)
        if model_type == "SDXL":
            pipeline.text_encoder_two.to(accelerator.device, dtype=tenc_dtype)

        # Get the models ready for training
        # TODO: Add logic to restore correct state according to model type
        # and tenc_training state
        pipeline.unet.train()
        pipeline.text_encoder.train()
        if model_type == "SDXL":
            pipeline.text_encoder_two.train()
        pipeline.vae.train()

        return pipeline
        

    def save_pipeline(self, pipeline=None):
        """
        Save the currrent pipeline state to lora and/or checkpoint
        """
        #pipeline = self.prepare_pipeline_for_inference(self.pipeline) \
        #    if pipeline is None or self.prepare_pipeline_for_inference(pipeline)

        model_type = self.model_type
        save_lora = self.save_lora
        save_checkpoint = self.save_checkpoint
        save_diffusers = self.save_difusers

        success = True
        # For all save types do the saving
        if save_lora:
            try:
                # Do lora stuff
                if model_type == "SDXL":
                    # Do SDXL Stuff for lora
            except:
                print("Error saving lora")
                success = False
        
        if save_checkpoint:
            try:
                # Do checkpoint stuff
                if model_type == "SDXL":
                # Do SDXL Stuff for checkpoints
            except:
                print("Error saving checkpoints")
                success = False

        if save_diffusers:
            try:
                # Do difuser stuff
                if model_type == "SDXL":
                # Do SDXL Stuff for difusers
            except:
                print("Error saving difusers")
                # Keep track of error and continue
                success
                
        # We done saving
        # Get the pipeline ready to resume training
        self.pipeline = self.prepare_pipeline_for_training(pipeline)
        # return false if anything went wrong so we can handle it externally
        return success
        
    
    def save_samples(self, pipeline):
        """
        Save sample images using current pipeline
        """
        pipeline = self.pipeline or pipeline
        # Get the pipeline ready for inference
        pipeline = self.prepare_pipeline_for_inference(pipeline)
        
        success = True
        try:
            # Image inference stuff goes here
            images = pipeline.save_image()
        except:
            print("Error saving image")
            success = False

        # Get the pipeline ready to resume training
        self.pipeline = self.prepare_pipeline_for_training(pipeline)
        return images or success

    def save_training_state(self, accelerator=None, pipeline=None, model_path=None, model_config=None):
        accelerator = self.accelerator if accelerator is None else accelerator
        model_path = self.model_path
        pipeline = self.pipeline if pipeline is None else pipeline
        model_config = self.model_config if model_config is None else model_config
        
        # Save the optimizer and scheduler states to model_path
        accelerator.save(self.pipeline.unet.optimizer.state_dict(), f"{model_path}/optimizer.pt")
        accelerator.save(self.pipeline.unet.scheduler.state_dict(), f"{model_path}/scheduler.pt")
        # Save the accelerator state to model_path
        accelerator.save(accelerator.state_dict(), f"{model_path}/accelerator.pt")
        # Save the pipeline state to model_path
        accelerator.save(self.pipeline.state_dict(), f"{model_path}/pipeline.pt")
        # Save the model config to model_path
        accelerator.save(shared.model_config, f"{model_path}/model_config.pt")
        return True

    def load_training_state(self, accelerator=None, pipeline=None, model_path=None, model_config=None):
        accelerator = self.accelerator if accelerator is None else accelerator
        model_path = self.model_path if model_path is None else model_path
        pipeline = self.pipeline if pipeline is None else pipeline
        model_config = self.model_config if model_config is None else model_config

        # Load all the .pt files located in model_path back into their relevant objects
        accelerator.
        optimizer.load_state_dict(accelerator.load(f"{model_path}/optimizer.pt"))
        pipeline.unet.scheduler.load_state_dict(accelerator.load(f"{model_path}/scheduler.pt"))
        accelerator.load_state_dict(accelerator.load(f"{model_path}/accelerator.pt"))
        pipeline.load_state_dict(accelerator.load(f"{model_path}/pipeline.pt"))
        model_config = accelerator.load(f"{model_path}/model_config.pt")

        return True

       


