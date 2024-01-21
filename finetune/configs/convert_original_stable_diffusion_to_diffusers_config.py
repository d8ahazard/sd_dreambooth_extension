from pydantic import Field
from finetune.dataclasses.base_config import BaseConfig


class ConvertOriginalStableDiffusionToDiffusersConfig(BaseConfig):
    checkpoint_path: str = Field(default=None, title='Checkpoint Path', description='Path to the checkpoint to convert.')
    clip_stats_path: str = Field(default=None, title='Clip Stats Path', description="Path to the clip stats file. Only required if the stable unclip model's config specifies `model.params.noise_aug_config.params.clip_stats_path`.")
    config_files: str = Field(default=None, title='Config Files', description='The YAML config file corresponding to the architecture.')
    controlnet: str = Field(default=None, title='Controlnet', description='Set flag if this is a controlnet checkpoint.')
    device: str = Field(default=None, title='Device', description='Device to use (e.g. cpu, cuda:0, cuda:1, etc.)')
    dump_path: str = Field(default=None, title='Dump Path', description='Path to the output model.')
    extract_ema: bool = Field(default=False, title='Extract Ema', description='Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning.')
    from_safetensors: bool = Field(default=False, title='From Safetensors', description='If `--checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.')
    half: bool = Field(default=False, title='Half', description='Save weights in half precision.')
    image_size: int = Field(default=512, title='Image Size', description='The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Siffusion v2 Base. Use 768 for Stable Diffusion v2.')
    num_in_channels: str = Field(default=None, title='Num In Channels', description='The number of input channels. If `None` number of input channels will be automatically inferred.')
    original_config_file: str = Field(default=None, title='Original Config File', description='The YAML config file corresponding to the original architecture.')
    pipeline_class_name: str = Field(default=None, title='Pipeline Class Name', description='Specify the pipeline class name')
    pipeline_type: str = Field(default=None, title='Pipeline Type', description="The pipeline type. One of 'FrozenOpenCLIPEmbedder', 'FrozenCLIPEmbedder', 'PaintByExample'. If `None` pipeline will be automatically inferred.")
    prediction_type: str = Field(default=None, title='Prediction Type', description="The prediction type that the model was trained on. Use 'epsilon' for Stable Diffusion v1.X and Stable Diffusion v2 Base. Use 'v_prediction' for Stable Diffusion v2.", choices=['epsilon', 'v_prediction'], group='Performance', advanced=True)
    scheduler_type: str = Field(default='pndm', title='Scheduler Type', description="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']")
    stable_unclip: str = Field(default=None, title='Stable Unclip', description="Set if this is a stable unCLIP model. One of 'txt2img' or 'img2img'.")
    stable_unclip_prior: str = Field(default=None, title='Stable Unclip Prior', description='Set if this is a stable unCLIP txt2img model. Selects which prior to use. If `--stable_unclip` is set to `txt2img`, the karlo prior (https://huggingface.co/kakaobrain/karlo-v1-alpha/tree/main/prior) is selected by default.')
    to_safetensors: bool = Field(default=True, title='To Safetensors', description='Whether to store pipeline in safetensors format or not.')
    upcast_attention: bool = Field(default=False, title='Upcast Attention', description='Whether the attention computation should always be upcasted. This is necessary when running stable diffusion 2.1.')
    vae_path: str = Field(default=None, title='Vae Path', description='Set to a path, hub id to an already converted vae to not convert it again.')
