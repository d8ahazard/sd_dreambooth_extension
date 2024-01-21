from pydantic import Field
from finetune.dataclasses.base_config import BaseConfig


class ConvertLoraSafetensorToDiffusersConfig(BaseConfig):
    alpha: float = Field(default=0.75, title='Alpha', description='The merging ratio in W = W0 + alpha * deltaW')
    base_model_path: str = Field(default=None, title='Base Model Path', description='Path to the base model in diffusers format.')
    checkpoint_path: str = Field(default=None, title='Checkpoint Path', description='Path to the checkpoint to convert.')
    device: str = Field(default=None, title='Device', description='Device to use (e.g. cpu, cuda:0, cuda:1, etc.)')
    dump_path: str = Field(default=None, title='Dump Path', description='Path to the output model.')
    lora_prefix_text_encoder: str = Field(default='lora_te', title='Lora Prefix Text Encoder', description='The prefix of text encoder weight in safetensors')
    lora_prefix_unet: str = Field(default='lora_unet', title='Lora Prefix Unet', description='The prefix of UNet weight in safetensors')
    to_safetensors: bool = Field(default=True, title='To Safetensors', description='Whether to store pipeline in safetensors format or not.')
