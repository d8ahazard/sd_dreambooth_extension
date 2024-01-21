from pydantic import Field
from finetune.dataclasses.base_config import BaseConfig


class ConvertDiffusersToOriginalSdxlConfig(BaseConfig):
    checkpoint_path: str = Field(default=None, title='Checkpoint Path', description='Path to the output model.')
    half: bool = Field(default=False, title='Half', description='Save weights in half precision.')
    model_path: str = Field(default=None, title='Model Path', description='Path to the model to convert.')
    use_safetensors: bool = Field(default=True, title='Use Safetensors', description='Save weights use safetensors, default is ckpt.')
