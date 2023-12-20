import asyncio
import logging
import os
import shutil
import tempfile
import traceback
from binascii import crc32
from concurrent.futures import ThreadPoolExecutor

import git
import requests
from fastapi import HTTPException
from huggingface_hub import snapshot_download

from finetune.helpers.ft_state import FinetuneStatus
from finetune.helpers.model_paths import MODEL_DIRS, BASE_MODELS
from helpers.download_progress import DownloadProgress

logger = logging.getLogger(__name__)


def download_checkpoint(new_model_url, model_type, token: str = None):
    if "civitai" in new_model_url:
        return download_civitai(new_model_url, model_type)
    else:
        return download_huggingface(new_model_url, model_type, token)


async def download_civitai(model_id=None, model_version_id=None):
    s_task = FinetuneStatus()
    model_json, version_json = fetch_civitai_info(model_id, model_version_id)
    if model_json is None or version_json is None:
        return
    model_id = model_json["id"]
    model_type = model_json["type"]
    model_files = version_json["files"]
    s_task.status = f"Downloading model {model_json['name']} from Civitai"

    # Get the primary model file or default to the first file if not found
    model_file = next((f for f in model_files if f.get('primary', False) is True),
                      model_files[0] if model_files else None)

    if not model_file:
        s_task.status = f"Error: No primary model found for MODEL_ID {model_id}."
        return
    download_url = model_file["downloadUrl"]

    # Get the base model type
    model_base = version_json["baseModel"]
    user_dir = MODEL_DIRS[model_type]

    # Set output directories based on the model type
    if model_type in ["Checkpoint", "LORA", "LoCon", "Hypernetwork", "TextualInversion", "Workflows", "Poses", "VAE",
                      "Controlnet"]:
        if model_base not in BASE_MODELS and model_base != "Other":
            s_task.status = f"Error: Unknown base model {model_base}."
            return
        user_dir = os.path.join(user_dir, BASE_MODELS[model_base])

    # Check if the model exists already.
    model_name = model_file['name']
    try:
        # Download the file
        logger.debug(f"Downloading {model_name} from {download_url}")
        model_file = os.path.join(user_dir, model_name)
        if not os.path.exists(model_file):
            logger.debug("No, really, downllading the model file.")
            await multipart_download(download_url, model_file)
        else:
            logger.info(f"Model {model_file} already exists. Skipping download.")

        s_task.status = f"Successfully downloaded {model_name} to {model_file}."
    except Exception as e:
        logger.error(f"Exception fetching model: {model_id}: {e}.")
        traceback.print_exc()
        s_task.status = f"Exception fetching model: {model_id}: {e}."
    return model_file


async def download_huggingface(model_path: str, model_type=None, hf_token=None):
    task_error_description = f"Downloading model from HuggingFace"
    s_task = FinetuneStatus()

    model_json, is_single_file = fetch_hf_info(model_path, model_type)
    logger.debug(f"Model JSON: {model_json}")
    dest = MODEL_DIRS[model_type]

    # Check if the model exists already.
    model_name = model_json['name']

    # Set up our status task
    logger.debug("Starting task...")
    s_task.status = task_error_description

    error_message = None
    if is_single_file:
        try:
            # The file name should be the last part of the model path
            model_file = os.path.join(dest, os.path.basename(model_path))
            await multipart_download(model_path, model_file)
            dest = model_file
        except Exception as e:
            logger.error(f"Exception fetching model: {model_path}: {e}.")
            error_message = f"Exception fetching model: {model_path}: {e}."
            dest = None
    else:
        try:
            temp_path = tempfile.mkdtemp()
            repo = git.Repo.clone_from(model_path, to_path=temp_path, no_checkout=True)
            file_list = [item.path for item in repo.head.commit.tree.traverse() if item.type == 'blob']
            logger.debug(f"File list: {file_list}")
            safetensors_files = [file for file in file_list if file.endswith('.safetensors')]
            ignore_patterns = ["README", ".gitattributes", "*.ckpt"] if safetensors_files else []
            cache_dir = "/tmp/dl_cache"
            os.makedirs(cache_dir, exist_ok=True)
            snapshot_download(repo_id=model_path, dest=dest, cache_dir=cache_dir,
                              force_download=True, token=hf_token if hf_token else None,
                              local_dir_use_symlinks=False,
                              ignore_patterns=ignore_patterns)
            shutil.rmtree(temp_path)
            dest = check_diffusers(dest)
        except Exception as e:
            error_message = f"Exception fetching model: {model_path}: {e}"
            dest = None
    if not error_message:
        s_task.status = f"Successfully downloaded {model_name} to {dest}."
        return dest
    else:
        logger.error(error_message)
        s_task.status = error_message
        return None


def get_model_by_id(model_id):
    response = requests.get(f"https://civitai.com/api/v1/models/{model_id}")
    if response.status_code != 200:
        return None
    return response.json()


def check_diffusers(model_path):
    diffusers_dirs = ["scheduler", "text_encoder", "tokenizer", "unet", "vae"]
    sdxl_dirs = ["text_encoder_2", "tokenizer_2"]
    diffusers_conf = "model_index.json"
    # Check if model_path has our diffusers dirs
    if not os.path.exists(model_path):
        return model_path

    for chk_dir in diffusers_dirs:
        if not os.path.exists(os.path.join(model_path, chk_dir)):
            print(f"Model path {model_path} does not contain {chk_dir}.")
            return model_path
    diffusers_model_folder = os.path.join(model_path, "diffusers", os.path.basename(model_path))
    if not os.path.exists(diffusers_model_folder):
        os.makedirs(diffusers_model_folder)
    for chk_dir in diffusers_dirs:
        print(f"Moving {chk_dir} to {diffusers_model_folder}")
        shutil.move(os.path.join(model_path, chk_dir), os.path.join(diffusers_model_folder, chk_dir))
    for chk_dir in sdxl_dirs:
        print(f"Moving {chk_dir} to {diffusers_model_folder}")
        if os.path.exists(os.path.join(model_path, chk_dir)):
            shutil.move(os.path.join(model_path, chk_dir), os.path.join(diffusers_model_folder, chk_dir))
    if os.path.exists(os.path.join(model_path, diffusers_conf)):
        print(f"Moving {diffusers_conf} to {diffusers_model_folder}")
        shutil.move(os.path.join(model_path, diffusers_conf), os.path.join(diffusers_model_folder, diffusers_conf))
    shutil.rmtree(model_path)
    return diffusers_model_folder


def get_model_by_version_id(model_version_id):
    response = requests.get(f"https://civitai.com/api/v1/model-versions/{model_version_id}")
    if response.status_code != 200:
        return None
    return response.json()


def get_model_by_hash(model_hash):
    response = requests.get(f"https://civitai.com/api/v1/models?hash={model_hash}")
    if response.status_code != 200:
        return None
    return response.json()


async def multipart_download(src_url, dest):
    num_parts = 4
    max_threads = 8
    loop = asyncio.get_event_loop()

    # Use loop.run_in_executor to run the blocking call in ThreadPoolExecutor
    size_res = await loop.run_in_executor(None, lambda: requests.get(src_url, stream=True, allow_redirects=True))

    total_size = int(size_res.headers.get('content-length', 0))
    # If total_size is greater than 2GB, set num_parts to 8
    # if total_size > 2147483648:
    #     num_parts = 8
    #     max_threads = 16
    part_size = total_size // num_parts

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as f:
        f.seek(total_size - 1)
        f.write(b'\0')

    parts = []
    for part in range(num_parts):
        logger.debug(f"Downloading part {part} of {num_parts}")
        start_byte = part * part_size
        if part == num_parts - 1:  # This is the last part
            end_byte = total_size - 1
        else:
            end_byte = (part + 1) * part_size - 1
        parts.append((src_url, dest, start_byte, end_byte))

    progress = DownloadProgress(total_size)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Using asyncio.gather to run all tasks concurrently
        await asyncio.gather(*[loop.run_in_executor(executor, download_part, part, progress) for part in parts])


def download_part(p_args, progress):
    src_url, dest, start_byte, end_byte = p_args
    headers = {
        "Range": f"bytes={start_byte}-{end_byte}"
    }
    part_response = requests.get(src_url, headers=headers, stream=True)
    with open(dest, 'r+b') as f:
        f.seek(start_byte)
        for p_chunk in part_response.iter_content(chunk_size=8192):
            f.write(p_chunk)
            progress.update(len(p_chunk))


def fetch_civitai_info(model_id, model_version_id):
    # If we have a version ID, just get that
    s_task = FinetuneStatus()
    version_json = None
    task_error_description = f"Download model from Civitai"
    if model_version_id is not None:
        version_json = get_model_by_version_id(model_version_id)
        response_model_id = version_json.get("modelId", None)
        if not response_model_id or response_model_id != model_id:
            s_task.status = f"Error: Model ID Mismatch. This shouldn't happen."
            return None, None

    # Retrieve this for storage
    model_json = get_model_by_id(model_id)
    if model_json is None:
        s_task.status = f"Error: Failed to fetch data for MODEL_ID {model_id}."
        return None, None

    # Extract the model type and name
    model_type = model_json["type"]
    model_name = model_json["name"]

    # Set a generic description
    task_error_description = f"Download model {model_name} from Civitai"

    # Check if the model type is supported
    if model_type not in MODEL_DIRS:
        s_task.status = f"Error: Unknown model type {model_type}."
        return None, None

    # Get the model versions if we haven't gotten it via the version ID
    if not version_json:
        if "modelVersions" not in model_json:
            s_task.status = f"Error: No model versions found for MODEL_ID {model_id}."
            return None, None

        model_versions = model_json["modelVersions"]
        if len(model_versions) == 0:
            s_task.status = f"Error: No model versions found for MODEL_ID {model_id}."
            return None, None
        # Select the most recent one
        version_json = model_versions[0]

    # Get the model files
    if "files" not in version_json:
        s_task.status = f"Error: No files found for MODEL_ID {model_id}."
        return None, None

    model_files = version_json["files"]
    if len(model_files) == 0:
        s_task.status = f"Error: No files found for MODEL_ID {model_id}."
        return None, None

    return model_json, version_json


def fetch_hf_info(model_path, model_type):
    # If the model_path is a full URL, check if it's a single file download or a full repo
    short_path = model_path.replace("https://", "")
    short_path = short_path.replace("http://", "")
    short_path = short_path.replace("huggingface.co/", "")
    short_path = short_path.split("/")
    # Retrieve the first two elements of the path
    if len(short_path) >= 2:
        # Model author is the first element of the path
        model_author = short_path[0]
        # Model name is the second element of the path
        model_name = short_path[1]
        short_path = "/".join(short_path[:2])
    else:
        raise HTTPException(status_code=400, detail="Invalid model path")
    # Verify that the readme exists at /raw/main/README.md
    logger.debug(f"Fetching readme from {short_path}")
    readme_url = f"https://huggingface.com/{short_path}/raw/main/README.md"
    response = requests.get(readme_url)
    readme_data = None
    if response.status_code != 200:
        base_url = f"https://huggingface.co/{short_path}"
        logger.debug(f"No readme found at {readme_url}, checking for repo at {base_url}")
        response = requests.get(base_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Unable to find model readme at {readme_url}")
    else:
        logger.debug(f"Found readme at {readme_url}")
        readme_data = response.text

    if "huggingface.co" in model_path:
        logger.debug(f"Model path is a HuggingFace URL: {model_path}")
        if "/blob/" in model_path or "/raw/" in model_path:
            model_path = model_path.replace("/blob/", "/resolve/")
            model_path = model_path.replace("/raw/", "/resolve/")
        if "/resolve/" in model_path:
            logger.debug(f"Model path is a single file download: {model_path}")
            is_single_file = True
        else:
            logger.debug(f"Model path is a full repo download: {model_path}")
            is_single_file = False
    else:
        model_path = short_path
        logger.debug(f"Model path is a 'short' Hub URL': {model_path}")
        is_single_file = True

    logger.debug(f"Model path is {model_path}")
    # If it's a single file, do a multipart download
    hash_id = crc32(short_path.encode())
    logger.debug(f"Hash ID is {hash_id}")
    model_json = {
        "name": model_name,
        "author": model_author,
        "tags": [],
        "images": [],
        "nsfw": False,
        "id": hash_id,
        "revisionId": hash_id,
        "downloadUrl": model_path,
        "type": model_type,
        "shortPath": short_path,
        "description": readme_data
    }
    logger.debug(f"Model JSON: {model_json}")
    return model_json, is_single_file
