import os
import traceback

import torch

from modules import shared
from modules.sd_models import CheckpointInfo

CONVERSION_MAP = {
    'v1x': ['convert_diffusers_to_original_stable_diffusion', 'convert_original_stable_diffusion_to_diffusers'],
    'v2x': ['convert_diffusers_to_original_stable_diffusion', 'convert_original_stable_diffusion_to_diffusers'],
    'v2x-512': ['convert_diffusers_to_original_stable_diffusion', 'convert_original_stable_diffusion_to_diffusers'],
    'sdxl': ['convert_diffusers_to_original_sdxl', 'convert_original_stable_diffusion_to_diffusers'],
    'lora': ['', 'convert_original_controlnet_to_diffusers'],
    'controlnet': ['', 'convert_original_controlnet_to_diffusers']
}

configs_path = os.path.join(shared.script_path, "extensions", "sd_dreambooth_extension", "configs")
CONFIG_MAP = {
    'v1x': os.path.join(configs_path, "v1-training-unfrozen.yaml"),
    'v2x': os.path.join(configs_path, "v2-training-unfrozen-v.yaml"),
    'v2x-512': os.path.join(configs_path, "v2-training-unfrozen.yaml"),
    'sdxl': os.path.join(configs_path, "sdxl-training-unfrozen.yaml"),
}

EXTRACT_PATH_MAP = {
    'v1x': os.path.join(shared.models_path, "diffusers", "v1x"),
    'v2x': os.path.join(shared.models_path, "diffusers", "v2x"),
    'v2x-512': os.path.join(shared.models_path, "diffusers", "v2x-512"),
    'sdxl': os.path.join(shared.models_path, "diffusers", "sdxl"),
    'lora': os.path.join(shared.models_path, "diffusers", "lora"),
    'controlnet': os.path.join(shared.models_path, "diffusers", "controlnet")
}

COMPILE_PATH_MAP = {
    'v1x': os.path.join(shared.models_path, "Stable-diffusion"),
    'v2x': os.path.join(shared.models_path, "Stable-diffusion"),
    'v2x-512': os.path.join(shared.models_path, "Stable-diffusion"),
    'sdxl': os.path.join(shared.models_path, "Stable-diffusion"),
    'lora': os.path.join(shared.models_path, "Lora"),
    'controlnet': os.path.join(shared.models_path, "controlnet")
}

SIZE_MAP = {
    'v1x': 512,
    'v2x': 768,
    'v2x-512': 512,
    'sdxl': 1024,
}

PIPE_CLASS_MAP = {
    'v1x': "StableDiffusionPipeline",
    'v2x': "StableDiffusionPipeline",
    'v2x-512': "StableDiffusionPipeline",
    'sdxl': "StableDiffusionXLPipeline"
}


def sanitize_model_name(model_name):
    # Remove all non-path characters
    model_name = ''.join(c for c in model_name if c.isalnum() or c in ['.', '_', '-'])
    return model_name


def set_config_attr(config, attr, value):
    if hasattr(config, attr):
        setattr(config, attr, value)
    else:
        print(f'Warning: config {config} does not have attribute {attr}')


def extract_checkpoint(checkpoint_path, model_type, **kwargs):
    model_methods = CONVERSION_MAP.get(model_type, None)
    output_path = EXTRACT_PATH_MAP.get(model_type, None)
    compile_path = COMPILE_PATH_MAP.get(model_type, None)
    pipe_class = PIPE_CLASS_MAP.get(model_type, None)
    # if [...] after checkpoint_path, remove it
    if '[' in checkpoint_path:
        checkpoint_path = checkpoint_path.split('[')[0].strip()
    # TODO: Add a check for the compile path before extracting
    if not model_methods:
        raise ValueError(f'Invalid model type {model_type}')
    if not output_path:
        raise ValueError(f'Invalid output path {output_path} for model type {model_type}')
    if not compile_path:
        raise ValueError(f'Invalid compile path {compile_path} for model type {model_type}')
    checkpoint_path = os.path.join(compile_path, checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise ValueError(f'Invalid checkpoint path {checkpoint_path} for model type {model_type}')

    # Ensure the output path exists
    os.makedirs(output_path, exist_ok=True)
    model_name = os.path.basename(checkpoint_path)
    model_name = os.path.splitext(model_name)[0]
    model_name = sanitize_model_name(model_name)
    output_path = os.path.join(output_path, model_name)

    conversion_config_str = model_methods[1] + '_config'

    conversion_script_module = f"finetune.conversion.{model_methods[1]}"
    conversion_config_module = f"finetune.configs.{conversion_config_str}"
    conversion_config_class = conversion_config_str.replace('_', ' ').title().replace(' ', '')

    conversion_script = __import__(conversion_script_module, fromlist=[''])
    # Get the convert() method from the script
    conversion_method = getattr(conversion_script, 'convert')
    conversion_config_module = __import__(conversion_config_module, fromlist=[''])
    conversion_config_class = getattr(conversion_config_module, conversion_config_class)

    # Try to populate the config with the kwargs
    try:
        conversion_config = conversion_config_class(**kwargs)
    except Exception as e:
        raise ValueError(f'Invalid conversion config {kwargs}') from e

    existing_hash = None
    hash_path = os.path.join(output_path, f'{model_name}.sha256')
    if os.path.exists(output_path):
        # Look for a .sha256 file with the source hash
        if os.path.exists(hash_path):
            with open(hash_path, 'r') as f:
                existing_hash = f.read()

    checkpoint_info = CheckpointInfo(checkpoint_path)
    checkpoint_hash = checkpoint_info.calculate_shorthash()
    if existing_hash == checkpoint_hash:
        print(f'Checkpoint {checkpoint_path} already extracted to {output_path}')
        return output_path

    set_config_attr(conversion_config, 'checkpoint_path', checkpoint_path)
    set_config_attr(conversion_config, 'dump_path', output_path)
    set_config_attr(conversion_config, 'use_safetensors', True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_config_attr(conversion_config, 'device', device)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_config_attr(conversion_config, 'device', device)
    if '.safetensors' in checkpoint_path:
        set_config_attr(conversion_config, 'from_safetensors', True)
    set_config_attr(conversion_config, 'to_safetensors', True)
    set_config_attr(conversion_config, 'image_size', SIZE_MAP.get(model_type, 512))
    original_config = CONFIG_MAP.get(model_type, None)
    if original_config:
        set_config_attr(conversion_config, 'original_config_file', original_config)
    else:
        print(f'Warning: no original config file found for model type {model_type}')

    if pipe_class:
        set_config_attr(conversion_config, 'pipeline_class_name', pipe_class)

    print(f'Converting {checkpoint_path} to {model_type} from config {conversion_config}...')
    # Run the convert() method from the script with the config
    try:
        conversion_method(conversion_config)
        with open(hash_path, 'w') as f:
            f.write(checkpoint_hash)
        conversion_config_json_path = os.path.join(output_path, f'{model_name}.json')
        conversion_config_json = conversion_config.json()
        with open(conversion_config_json_path, 'w') as f:
            f.write(conversion_config_json)
    except Exception as e:
        print(f'Error converting {checkpoint_path} to {model_type} from config {conversion_config}: {e}')
        traceback.print_exc()


    return output_path


def compile_checkpoint(checkpoint_path, model_type, **kwargs):
    model_methods = CONVERSION_MAP.get(model_type, None)
    output_path = COMPILE_PATH_MAP.get(model_type, None)

    if not model_methods:
        raise ValueError(f'Invalid model type {model_type}')
    if not os.path.exists(checkpoint_path):
        raise ValueError(f'Invalid checkpoint path {checkpoint_path}')
    if not output_path:
        raise ValueError(f'Invalid output path {output_path}')

    # Ensure the output path exists
    os.makedirs(output_path, exist_ok=True)
    model_name = os.path.basename(checkpoint_path)
    model_name = os.path.splitext(model_name)[0]
    model_name = sanitize_model_name(model_name)
    output_path = os.path.join(output_path, model_name)

    conversion_config_str = model_methods[0] + '_config'

    conversion_script_module = f"finetune.conversion.{model_methods[1]}"
    conversion_config_module = f"finetune.configs.{conversion_config_str}"
    conversion_config_class = conversion_config_str.replace('_', '').title()

    conversion_script = __import__(conversion_script_module, fromlist=[''])
    # Get the convert() method from the script
    conversion_method = getattr(conversion_script, 'convert')
    conversion_config_module = __import__(conversion_config_module, fromlist=[''])
    conversion_config_class = getattr(conversion_config_module, conversion_config_class)

    # Try to populate the config with the kwargs
    try:
        conversion_config = conversion_config_class(**kwargs)
    except Exception as e:
        raise ValueError(f'Invalid conversion config {kwargs}') from e

    existing_hash = None
    hash_path = os.path.join(output_path, f'{model_name}.sha256')
    if os.path.exists(output_path):
        # Look for a .sha256 file with the source hash
        if os.path.exists(hash_path):
            with open(hash_path, 'r') as f:
                existing_hash = f.read()

    checkpoint_info = CheckpointInfo(checkpoint_path)
    checkpoint_hash = checkpoint_info.calculate_shorthash()
    if existing_hash == checkpoint_hash:
        print(f'Checkpoint {checkpoint_path} already compiled to {output_path}')
        return output_path

    set_config_attr(conversion_config, 'checkpoint_path', checkpoint_path)
    set_config_attr(conversion_config, 'dump_path', output_path)
    set_config_attr(conversion_config, 'use_safetensors', True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_config_attr(conversion_config, 'device', device)
    if '.safetensors' in checkpoint_path:
        set_config_attr(conversion_config, 'from_safetensors', True)
    set_config_attr(conversion_config, 'to_safetensors', True)
    set_config_attr(conversion_config, 'image_size', SIZE_MAP.get(model_type, 512))
    print(f'Converting {checkpoint_path} to {model_type} from config {conversion_config}...')
    try:
        conversion_method(conversion_config)
        with open(hash_path, 'w') as f:
            f.write(checkpoint_hash)
    except Exception as e:
        raise ValueError(f'Invalid conversion script {conversion_script_module}') from e
    return output_path
