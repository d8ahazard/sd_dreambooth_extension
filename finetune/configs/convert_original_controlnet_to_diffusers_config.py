from pydantic import Field
from finetune.dataclasses.base_config import BaseConfig


class ConvertOriginalControlnetToDiffusersConfig(BaseConfig):
    checkpoint_path: str = Field(default=None, title='Checkpoint Path', description='Path to the checkpoint to convert.')
    cross_attention_dim: str = Field(default=None, title='Cross Attention Dim', description='Override for cross attention_dim')
    device: str = Field(default=None, title='Device', description='Device to use (e.g. cpu, cuda:0, cuda:1, etc.)')
    dump_path: str = Field(default=None, title='Dump Path', description='Path to the output model.')
    extract_ema: bool = Field(default=False, title='Extract Ema', description='Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning.')
    from_safetensors: bool = Field(default=False, title='From Safetensors', description='If `--checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.')
    image_size: int = Field(default=512, title='Image Size', description='The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Siffusion v2 Base. Use 768 for Stable Diffusion v2.')
    num_in_channels: str = Field(default=None, title='Num In Channels', description='The number of input channels. If `None` number of input channels will be automatically inferred.')
    original_config_file: str = Field(default=None, title='Original Config File', description='The YAML config file corresponding to the original architecture.')
    to_safetensors: bool = Field(default=True, title='To Safetensors', description='Whether to store pipeline in safetensors format or not.')
    upcast_attention: bool = Field(default=False, title='Upcast Attention', description='Whether the attention computation should always be upcasted. This is necessary when running stable diffusion 2.1.')
    use_linear_projection: str = Field(default=None, title='Use Linear Projection', description='Override for use linear projection')
