import asyncio
import base64
import hashlib
import io
import json
import os
import traceback
import zipfile
from pathlib import Path

import gradio as gr
from PIL import Image
from fastapi import FastAPI, Response, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from pydantic.dataclasses import Union
from pydantic.types import List

import modules.script_callbacks as script_callbacks
from extensions.sd_dreambooth_extension.dreambooth import dream_state
from extensions.sd_dreambooth_extension.dreambooth.db_config import from_file, DreamboothConfig
from extensions.sd_dreambooth_extension.dreambooth.diff_to_sd import compile_checkpoint
from extensions.sd_dreambooth_extension.dreambooth.dream_state import DreamState
from extensions.sd_dreambooth_extension.dreambooth.finetune_utils import FilenameTextGetter
from extensions.sd_dreambooth_extension.dreambooth.sd_to_diff import extract_checkpoint
from extensions.sd_dreambooth_extension.dreambooth.secret import get_secret
from extensions.sd_dreambooth_extension.dreambooth.utils import wrap_gpu_call, get_images
from extensions.sd_dreambooth_extension.scripts import dreambooth
from extensions.sd_dreambooth_extension.scripts.dreambooth import ui_samples
from modules import shared


class InstanceData(BaseModel):
    data: str = Field(title="File data", description="Base64 representation of the file")
    name: str = Field(title="File name")
    txt: str = Field(title="Prompt")


class ImageData:
    def __init__(self, name, prompt, data):
        self.name = name
        self.prompt = prompt
        self.data = data

    def dict(self):
        return {
            "name": self.name,
            "data": self.data,
            "txt": self.prompt
        }


class DbImagesRequest(BaseModel):
    imageList: List[InstanceData] = Field(title="Images",
                                          description="List of images to work on. Must be Base64 strings")


class DreamboothConcept(BaseModel):
    max_steps: int = -1
    instance_data_dir: str = ""
    class_data_dir: str = ""
    file_prompt_contents: str = ""
    instance_prompt: str = ""
    class_prompt: Union[str, None] = ""
    save_sample_prompt: Union[str, None] = ""
    save_sample_template: Union[str, None] = ""
    instance_token: Union[str, None] = ""
    class_token: Union[str, None] = ""
    num_class_images: int = 0
    class_negative_prompt: Union[str, None] = ""
    class_guidance_scale: float = 7.5
    class_infer_steps: int = 60
    save_sample_negative_prompt: Union[str, None] = ""
    n_save_sample: int = 1
    sample_seed: int = -1
    save_guidance_scale: float = 7.5
    save_infer_steps: int = 60


class DreamboothParameters(BaseModel):
    concepts_list: list[DreamboothConcept]
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adam_weight_decay: float = 0.01
    attention: str = "default"
    center_crop: bool = False
    concepts_path: Union[str, None] = ""
    custom_model_name: Union[str, None] = ""
    epoch_pause_frequency: int = 0
    epoch_pause_time: int = 60
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    half_model: bool = False
    hflip: bool = True
    learning_rate: float = 0.000002
    lora_learning_rate: float = 0.0002
    lora_txt_learning_rate: float = 0.0002
    lora_txt_weight: int = 1
    lora_weight: int = 1
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    max_token_length: int = 75
    max_train_steps: int = 0
    mixed_precision: str = "no"
    model_dir: Union[str, None] = ""
    model_name: str = ""
    not_cache_latents: bool = True
    num_train_epochs: int = 100
    pad_tokens: bool = True
    pretrained_model_name_or_path: Union[str, None] = ""
    pretrained_vae_name_or_path: Union[str, None] = ""
    prior_loss_weight: int = 1
    resolution: int = 512
    revision: int = 0
    sample_batch_size: int = 1
    save_ckpt_after: bool = True
    save_ckpt_cancel: bool = True
    save_ckpt_during: bool = True
    save_class_txt: bool = True
    save_embedding_every: int = 500
    save_lora_after: bool = True
    save_lora_cancel: bool = True
    save_lora_during: bool = True
    save_preview_every: int = 500
    save_state_after: bool = True
    save_state_cancel: bool = False
    save_state_during: bool = False
    save_use_global_counts: bool = True
    save_use_epochs: bool = False
    scale_lr: bool = False
    src: Union[str, None] = ""
    shuffle_tags: bool = False
    train_batch_size: int = 1
    train_text_encoder: bool = True
    use_8bit_adam: bool = False


def zip_files(db_model_name, files, name_part=""):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a",
                         zipfile.ZIP_DEFLATED, False) as zip_file:
        for file in files:
            if isinstance(file, str):
                print(f"Zipping img: {file}")
                if os.path.exists(file) and os.path.isfile(file):
                    parent_path = os.path.join(Path(file).parent), Path(file).name
                    zip_file.write(file, arcname=parent_path)
                    check_txt = os.path.join(os.path.splitext(file)[0], ".txt")
                    if os.path.exists(check_txt):
                        print(f"Zipping txt: {check_txt}")
                        parent_path = os.path.join(Path(check_txt).parent, Path(check_txt).name)
                        zip_file.write(check_txt, arcname=parent_path)
            else:
                img_byte_arr = io.BytesIO()
                file.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                file_name = hashlib.sha1(file.tobytes()).hexdigest()
                image_filename = f"{file_name}.png"
                zip_file.writestr(image_filename, img_byte_arr)
    zip_file.close()
    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment; filename={db_model_name}{name_part}_images.zip"}
    )


def check_api_key(key):
    current_key = get_secret()
    if current_key is not None and current_key != "":
        if key is None or key == "":
            return JSONResponse(status_code=401, content={"message": "API Key Required."})
        if key != current_key:
            return JSONResponse(status_code=403, content={"message": "Invalid API Key."})
    return None


def base64_to_pil(im_b64) -> Image:
    im_b64 = bytes(im_b64, 'utf-8')
    im_bytes = base64.b64decode(im_b64)  # im_bytes is a binary image
    im_file = io.BytesIO(im_bytes)  # convert image to file-like object
    img = Image.open(im_file)
    return img


def file_to_base64(file_path) -> str:
    with open(file_path, "rb") as f:
        im_b64 = base64.b64encode(f.read())
        return str(im_b64, 'utf-8')


def dreamBoothAPI(demo: gr.Blocks, app: FastAPI):
    @app.post("/dreambooth/createModel")
    async def create_model(
            new_model_name: str = Query(None, description="The name of the model to create.", ),
            new_model_src: str = Query(None, description="The source checkpoint to extract to create this model.", ),
            new_model_scheduler: str = Query(None, description="The scheduler to use. V2+ models ignore this.", ),
            create_from_hub: bool = Query(False, description="Create this model from the hub", ),
            new_model_url: str = Query(None,
                                       description="The hub URL to use for this model. Must contain diffusers model.", ),
            new_model_token: str = Query(None, description="Your huggingface hub token.", ),
            new_model_extract_ema: bool = Query(False, description="Whether to extract EMA weights if present.", ),
            api_key: str = Query("", description="If an API key is set, this must be present.", ),
    ):
        """
        Create a new Dreambooth model.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check

        print("Creating new Checkpoint: " + new_model_name)
        fn = extract_checkpoint(new_model_name,
                                new_model_src,
                                new_model_scheduler,
                                create_from_hub,
                                new_model_url,
                                new_model_token,
                                new_model_extract_ema)

    @app.post("/dreambooth/start_training")
    async def start_training(
            params: DreamboothParameters = Body(None,
                                                description="An optional json object containing training settings.", ),
            model_name: str = Query(None,
                                    description="Only used if params are not specified. The model name to load params for.", ),
            lora_model_name: str = Query("",
                                         description="The name of a lora weights file to train with. 'Use lora' should be enabled in params/saved settings.", ),
            lora_weight: float = Query(1.0,
                                       description="How strongly the lora UNET will be merged with the base model.", ),
            lora_txt_weight: float = Query(1.0,
                                           description="How strongly the lora Text Encoder will be merged with the base model.", ),
            use_imagic: bool = Query(False, description="Train using Imagic instead of 'traditional' Dreambooth.", ),
            use_subdir: bool = Query(False,
                                     description="Save checkpoints to a subdirectory in the base model directory.", ),
            custom_name: str = Query("",
                                     description="A custom name to use when saving checkpoints. Will also be used for the subdir if enabled.", ),
            use_tx2img: bool = Query(True, description="Use txt2img to generate class images.", ),
            api_key: str = Query("", description="If an API key is set, this must be present.", ),
    ):
        """
        Start training dreambooth.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        print("Starting Training")
        if params is not None:
            model_name = params.model_name
            lora_weight = params.lora_weight
            lora_txt_weight = params.lora_txt_weight

        task = asyncio.create_task(train_model(model_name, lora_model_name, lora_weight, lora_txt_weight, use_imagic,
                                               use_subdir, custom_name, use_tx2img))
        return {"status": "finished"}

    async def train_model(model_name,
                          lora_model_name,
                          lora_weight,
                          lora_txt_weight,
                          train_imagic_only,
                          use_subdir,
                          custom_model_name,
                          use_txt2img):

        wrap_gpu_call(dreambooth.start_training(
            model_name,
            lora_model_name,
            lora_weight,
            lora_txt_weight,
            train_imagic_only,
            use_subdir,
            custom_model_name,
            use_txt2img
        ))

    @app.get("/dreambooth/status")
    async def check_status(
            api_key: str = Query("", description="If an API key is set, this must be present.", )) -> DreamState:
        """
        Check the current state of Dreambooth processes.
        @return:
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        return {"current_state": f"{json.dumps(dream_state.status.dict())}"}

    @app.get("/dreambooth/model_config")
    async def get_model_config(
            model_name: str = Query(None, description="The model name to fetch config for."),
            api_key: str = Query("", description="If an API key is set, this must be present.", )
    ) -> DreamboothConfig:
        """
        Get a specified model config.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        cfg = from_file(model_name)
        if cfg:
            return JSONResponse(content=cfg.__dict__)
        return {"Exception": "Config not found."}

    @app.post("/dreambooth/model_config")
    async def set_model_config(
            model_cfg: DreamboothParameters = Body(description="The config to save"),
            api_key: str = Query("", description="If an API key is set, this must be present.", )
    ):
        """
        Save a model config from JSON.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        try:
            print("Create config")
            config = DreamboothConfig()
            for key in model_cfg.dict():
                if key in config.__dict__:
                    config.__dict__[key] = model_cfg.dict()[key]
            config.save()
            print("Saved?")
            return JSONResponse(content=config.__dict__)
        except Exception as e:
            traceback.print_exc()
            return {"Exception saving model": f"{e}"}

    @app.get("/dreambooth/get_checkpoint")
    async def get_checkpoint(
            model_name: str = Query(description="The model name of the checkpoint to get."),
            skip_build: bool = Query(True, description="Set to false to re-compile the checkpoint before retrieval."),
            lora_model_name: str = Query("",
                                         description="The (optional) name of the lora model to merge with the checkpoint."),
            save_model_name: str = Query("", description="A custom name to use when generating the checkpoint."),
            lora_weight: int = Query(1, description="The weight of the lora UNET when merged with the checkpoint."),
            lora_text_weight: int = Query(1,
                                          description="The weight of the lora Text Encoder when merged with the checkpoint."),
            api_key: str = Query("", description="If an API key is set, this must be present.", )
    ):
        """
        Generate and zip a checkpoint for a specified model.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        config = from_file(model_name)
        path = None
        if save_model_name == "" or save_model_name is None:
            save_model_name = model_name
        if skip_build:
            ckpt_dir = shared.cmd_opts.ckpt_dir
            models_path = os.path.join(shared.models_path, "Stable-diffusion")
            if ckpt_dir is not None:
                models_path = ckpt_dir
            use_subdir = False
            if "use_subdir" in config.__dict__:
                use_subdir = config["use_subdir"]
            total_steps = config.revision
            if use_subdir:
                checkpoint_path = os.path.join(models_path, save_model_name, f"{save_model_name}_{total_steps}.ckpt")
            else:
                checkpoint_path = os.path.join(models_path, f"{save_model_name}_{total_steps}.ckpt")
            print(f"Looking for checkpoint at {checkpoint_path}")
            if os.path.exists(checkpoint_path):
                print("Existing checkpoint found, returning.")
                path = checkpoint_path
            else:
                skip_build = False
        if not skip_build:
            ckpt_result = compile_checkpoint(model_name, config.half_model, False, lora_model_name, lora_weight,
                                             lora_text_weight, save_model_name, False, True)
            if "Checkpoint compiled successfully" in ckpt_result:
                path = ckpt_result.replace("Checkpoint compiled successfully:", "").strip()
                print(f"Checkpoint aved to path: {path}")

        if path is not None and os.path.exists(path):
            print(f"Returning file response: {path}-{os.path.splitext(path)}")
            return FileResponse(path)

        return {"exception": f"Unable to find or compile checkpoint."}

    @app.get("/dreambooth/samples")
    async def generate_samples(
            model_name: str = Query(description="The model name to use for generating samples."),
            sample_prompt: str = Query(description="The prompt to use to generate sample images."),
            num_images: int = Query(1, description="The number of sample images to generate."),
            batch_size: int = Query(1, description="How many images to generate at once."),
            lora_model_path: str = Query("", description="The path to a lora model to use when generating images."),
            lora_weight: float = Query(1.0, description="The weight of the lora unet when merging with the base model."),
            lora_txt_weight: float = Query(1.0, description="The weight of the lora text encoder when merging with the base model"),
            negative_prompt: str = Query("", description="An optional negative prompt to use when generating images."),
            seed: int = Query(-1, description="The seed to use when generating samples"),
            steps: int = Query(60, description="Number of sampling steps to use when generating images."),
            scale: float = Query(7.5, description="CFG scale to use when generating images."),
            api_key: str = Query("", description="If an API key is set, this must be present.", )
    ):
        """
        Generate sample images for a specified model.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        images, msg = ui_samples(
            model_dir=model_name,
            save_sample_prompt=sample_prompt,
            num_samples=num_images,
            batch_size=batch_size,
            lora_model_path=lora_model_path,
            lora_weight=lora_weight,
            lora_txt_weight=lora_txt_weight,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            scale=scale
        )
        if len(images) > 1:
            return zip_files(model_name, images, "_sample")
        else:
            img_byte_arr = io.BytesIO()
            file = images[0]
            file.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")

    @app.post("/dreambooth/classifiers")
    async def generate_classifiers(
            model_name: str = Query(description="The model name to generate classifiers for."),
            api_key: str = Query("", description="If an API key is set, this must be present.", )
    ):
        """
        Generate classification images for a model based on a saved config.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        _ = asyncio.create_task(generate_classifiers(model_name))
        return {f"Generating classification images..."}

    @app.get("/dreambooth/classifiers")
    async def get_classifiers(
            model_name: str = Query(description="The model name to retrieve classifiers for."),
            concept_idx: int = Query(-1, description="If set, will retrieve the specified concept's class images. Otherwise, all class images will be retrieved."),
            api_key: str = Query("", description="If an API key is set, this must be present.", )
    ):
        """
        Retrieve generated classifier images from a saved model config.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        config = from_file(model_name)
        concepts = config.concepts_list
        concept_dict = {}
        out_images = []
        if concept_idx >= 0:
            if len(concepts) - 1 >= concept_idx:
                print(f"Returning class images for concept {concept_idx}")
                concept_dict[concept_idx] = concepts[concept_idx]
            else:
                return {"Exception": f"Concept index {concept_idx} out of range."}
        else:
            c_idx = 0
            for concept in concepts:
                concept_dict[c_idx] = concept

        for concept_key in concept_dict:
            concept = concept_dict[concept_key]
            class_images_dir = concept["class_data_dir"]
            if class_images_dir == "" or class_images_dir is None or class_images_dir == shared.script_path:
                class_images_dir = os.path.join(config.model_dir, f"classifiers_{concept_key}")
                print(f"Class image dir is not set, defaulting to {class_images_dir}")
            if os.path.exists(class_images_dir):
                from extensions.sd_dreambooth_extension.dreambooth.utils import get_images
                class_images = get_images(class_images_dir)
                for image in class_images:
                    out_images.append(str(image))

        if len(out_images) > 0:
            return zip_files(model_name, out_images, "_class")
        else:
            return {"Result": "No images found."}

    @app.post("/dreambooth/upload")
    async def upload_db_images(
            model_name: str = Query(description="The model name to upload images for."),
            instance_name: str = Query(description="The concept/instance name the images are for."),
            images: DbImagesRequest = Body(description="A dictionary of images, filenames, and prompts to save."),
            api_key: str = Query("", description="If an API key is set, this must be present.", )
    ):
        """
        Upload images for training.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        
        root_img_path = os.path.join(shared.script_path, "..", "InstanceImages")
        if not os.path.exists(root_img_path):
            print(f"Creating root instance dir: {root_img_path}")
            os.makedirs(root_img_path)

        image_dir = os.path.join(root_img_path, model_name, instance_name)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        image_paths = []
        for img_data in images.imageList:
            img = base64_to_pil(img_data.data)
            name = img_data.name
            prompt = img_data.txt
            image_path = os.path.join(image_dir, name)
            text_path = os.path.splitext(image_path)[0]
            text_path = F"{text_path}.txt"
            print(f"Saving image to: {image_path}")
            img.save(image_path)
            print(f"Saving prompt to: {text_path}")
            with open(text_path, "w") as tx_file:
                tx_file.writelines(prompt)
            image_paths.append(image_path)

        return {"Status": f"Saved {len(image_paths)} images.", "Images": {x for x in image_paths}}

    @app.get("/dreambooth/testimg")
    async def generate_test_data():
        model_dir = "E:\\dev\\sd_db\\mj_5"
        text_getter = FilenameTextGetter(False)
        instance_images = get_images(model_dir)
        inst_datas = []
        for x in instance_images:
            image_bytes = file_to_base64(x)
            name = x.name + x.suffix
            txt = text_getter.read_text(x)
            inst_datas.append(ImageData(name, txt, image_bytes).dict())
        return JSONResponse(inst_datas)


script_callbacks.on_app_started(dreamBoothAPI)

print("Dreambooth API layer loaded")
