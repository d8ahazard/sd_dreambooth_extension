import base64
import functools
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

from extensions.sd_dreambooth_extension.dreambooth import db_shared
from extensions.sd_dreambooth_extension.dreambooth.db_config import from_file, DreamboothConfig
from extensions.sd_dreambooth_extension.dreambooth.db_shared import DreamState
from extensions.sd_dreambooth_extension.dreambooth.diff_to_sd import compile_checkpoint
from extensions.sd_dreambooth_extension.dreambooth.finetune_utils import FilenameTextGetter, generate_classifiers
from extensions.sd_dreambooth_extension.dreambooth.sd_to_diff import extract_checkpoint
from extensions.sd_dreambooth_extension.dreambooth.secret import get_secret
from extensions.sd_dreambooth_extension.dreambooth.utils import get_images
from extensions.sd_dreambooth_extension.scripts import dreambooth
from extensions.sd_dreambooth_extension.scripts.dreambooth import ui_samples
from modules import sd_models


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


# API Representation of concept data
class DreamboothConcept(BaseModel):
    instance_data_dir: str = ""
    class_data_dir: str = ""
    instance_prompt: str = ""
    class_prompt: Union[str, None] = ""
    save_sample_prompt: Union[str, None] = ""
    save_sample_template: Union[str, None] = ""
    instance_token: Union[str, None] = ""
    class_token: Union[str, None] = ""
    num_class_images_per: int = 0
    class_negative_prompt: Union[str, None] = ""
    class_guidance_scale: float = 7.5
    class_infer_steps: int = 60
    save_sample_negative_prompt: Union[str, None] = ""
    n_save_sample: int = 1
    sample_seed: int = -1
    save_guidance_scale: float = 7.5
    save_infer_steps: int = 60


# API Representation of db config
class DreamboothParameters(BaseModel):
    concepts_list: List[DreamboothConcept]
    attention: str = "default"
    cache_latents: bool = True
    center_crop: bool = False
    clip_skip: int = 1
    concepts_path: Union[str, None] = ""
    custom_model_name: Union[str, None] = ""
    epoch_pause_frequency: int = 0
    epoch_pause_time: int = 60
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    gradient_set_to_none: bool = True
    graph_smoothing: int = 50
    half_model: bool = False
    hflip: bool = True
    learning_rate: float = 0.000002
    learning_rate_min: float = 0.000001
    lora_learning_rate: float = 0.0002
    lora_model_name: str = ""
    lora_rank: int = 4
    lora_txt_learning_rate: float = 0.0002
    lora_txt_weight: int = 1
    lora_weight: int = 1
    lr_cycles: int = 1
    lr_factor: float = 0.5
    lr_power: float = 1.0
    lr_scale_pos: float = 0.5
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    max_token_length: int = 75
    mixed_precision: str = "no"
    adamw_weight_decay: float = 0.01
    model_name: str = ""
    num_train_epochs: int = 100
    pad_tokens: bool = True
    pretrained_vae_name_or_path: Union[str, None] = ""
    prior_loss_scale: bool = False
    prior_loss_target: int = 100
    prior_loss_weight: float = 1.0
    prior_loss_weight_min: float = 0.1
    resolution: int = 512
    revision: int = 0
    sample_batch_size: int = 1
    sanity_prompt: str = ""
    sanity_seed: int = 420420
    save_ckpt_after: bool = True
    save_ckpt_cancel: bool = False
    save_ckpt_during: bool = True
    save_embedding_every: int = 25
    save_lora_after: bool = True
    save_lora_cancel: bool = False
    save_lora_during: bool = True
    save_preview_every: int = 5
    save_safetensors: bool = False
    save_state_after: bool = False
    save_state_cancel: bool = False
    save_state_during: bool = False
    snapshot: str = False
    src: Union[str, None] = ""
    shuffle_tags: bool = False
    train_batch_size: int = 1
    train_imagic: bool = False
    stop_text_encoder: float = 0
    use_8bit_adam: bool = False
    use_concepts: bool = False
    use_ema: bool = True
    use_lora: bool = False
    use_subdir: bool = True


import asyncio

def run_in_background(func, *args, **kwargs):
    """
    Wrapper function to run a non-asynchronous method as a task in the event loop.
    """
    async def wrapper():
        new_func = functools.partial(func, *args, **kwargs)
        await asyncio.get_running_loop().run_in_executor(None, new_func)
    asyncio.create_task(wrapper())
def zip_files(db_model_name, files, name_part=""):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a",
                         zipfile.ZIP_DEFLATED, False) as zip_file:
        for file in files:
            if isinstance(file, str):
                print(f"Zipping img: {file}")
                if os.path.exists(file) and os.path.isfile(file):
                    parent_path = os.path.join(Path(file).parent, Path(file).name)
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


def dreambooth_api(_: gr.Blocks, app: FastAPI):
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

        if new_model_name is None or new_model_name == "":
            return JSONResponse(status_code=422, content={"message": "Invalid model name."})

        if db_shared.status.job_count != 0:
            print("Something is already running.")
            return JSONResponse(content={"message": "Job already in progress.", "status": db_shared.status.dict()})


        print("Creating new Checkpoint: " + new_model_name)
        _ = extract_checkpoint(new_model_name,
                               new_model_src,
                               new_model_scheduler,
                               create_from_hub,
                               new_model_url,
                               new_model_token,
                               new_model_extract_ema)

    @app.post("/dreambooth/start_training")
    async def start_training(
            model_name: str = Query(None,
                                    description="The model name to load params for.", ),
            use_tx2img: bool = Query(True, description="Use txt2img to generate class images.", ),
            api_key: str = Query("", description="If an API key is set, this must be present.", ),
    ):
        """
        Start training dreambooth.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check

        if model_name is None or model_name == "":
            return JSONResponse(status_code=422, content={"message": "Invalid model name."})
        config = from_file(model_name)
        if config is None:
            return JSONResponse(status_code=422, content={"message": "Invalid config."})
        if db_shared.status.job_count != 0:
            print("Something is already running.")
            return JSONResponse(content={"message": "Job already in progress.", "status": db_shared.status.dict()})

        print("Starting Training")
        if db_shared.status.job_count != 0:
            print("Something is already running.")
            return JSONResponse(content={"message": "Job already in progress.", "status": db_shared.status.dict()})
        db_shared.status.begin()
        run_in_background(dreambooth.start_training, model_name, use_tx2img)
        return {"Status": "Training started."}

    @app.get("/dreambooth/cancel")
    async def cancel_jobs(
            api_key: str = Query("", description="If an API key is set, this must be present.", )) -> \
            Union[DreamState, JSONResponse]:
        """
        Check the current state of Dreambooth processes.
        @return:
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        if db_shared.status.job_count == 0:
            return JSONResponse(content={"message": "Nothing to cancel."})
        db_shared.status.interrupted = True
        return JSONResponse(content={"message": f"Processes cancelled."})

    @app.get("/dreambooth/status")
    async def check_status(
            api_key: str = Query("", description="If an API key is set, this must be present.", )) -> \
            Union[DreamState, JSONResponse]:
        """
        Check the current state of Dreambooth processes.
        @return:
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        return JSONResponse(content={"current_state": f"{json.dumps(db_shared.status.dict())}"})

    @app.get("/dreambooth/status_images")
    async def check_status_images(
            api_key: str = Query("", description="If an API key is set, this must be present.", )) -> JSONResponse:
        """
        Retrieve any images that may currently be present in the state.
        Args:
            api_key: An API key, if one has been set in the UI.

        Returns:
            A single image or zip of images, depending on how many exist.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        db_shared.status.set_current_image()
        images = db_shared.status.current_image
        if not isinstance(images, List):
            if images is not None:
                images = [images]
            else:
                images = []
        if len(images) == 0:
            return JSONResponse(content={"message": "No images."})
        if len(images) > 1:
            return zip_files("status", images, "_sample")
        else:
            file = images[0]
            if isinstance(file, str):
                file = Image.open(file)
            img_byte_arr = io.BytesIO()
            file.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")

    @app.get("/dreambooth/model_config")
    async def get_model_config(
            model_name: str = Query(None, description="The model name to fetch config for."),
            api_key: str = Query("", description="If an API key is set, this must be present.", )
    ) -> Union[DreamboothConfig, JSONResponse]:
        """
        Get a specified model config.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        if model_name is None or model_name == "":
            return JSONResponse(status_code=422, content={"message": "Invalid model name."})
        config = from_file(model_name)
        if config is None:
            return JSONResponse(status_code=422, content={"message": "Invalid config."})
        if db_shared.status.job_count != 0:
            print("Something is already running.")
            return JSONResponse(content={"message": "Job already in progress.", "status": db_shared.status.dict()})

        return JSONResponse(content=config.__dict__)

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
            api_key: str = Query("", description="If an API key is set, this must be present.", )
    ):
        """
        Generate and zip a checkpoint for a specified model.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        if model_name is None or model_name == "":
            return JSONResponse(status_code=422, content={"message": "Invalid model name."})
        config = from_file(model_name)
        if config is None:
            return JSONResponse(status_code=422, content={"message": "Invalid config."})
        if db_shared.status.job_count != 0:
            print("Something is already running.")
            return JSONResponse(content={"message": "Job already in progress.", "status": db_shared.status.dict()})
        path = None
        save_model_name = config.model_name
        if config.custom_model_name:
            save_model_name = config.custom_model_name
        if skip_build:
            ckpt_dir = db_shared.ckpt_dir
            models_path = os.path.join(db_shared.models_path, "Stable-diffusion")
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
            if config.save_safetensors:
                checkpoint_path = checkpoint_path.replace(".ckpt", ".safetensors")
            print(f"Looking for checkpoint at {checkpoint_path}")
            if os.path.exists(checkpoint_path):
                print("Existing checkpoint found, returning.")
                path = checkpoint_path
            else:
                skip_build = False
        if not skip_build:
            db_shared.status.begin()
            ckpt_result = compile_checkpoint(model_name, reload_models=False, log=False)
            db_shared.status.end()
            if "Checkpoint compiled successfully" in ckpt_result:
                path = ckpt_result.replace("Checkpoint compiled successfully:", "").strip()
                print(f"Checkpoint aved to path: {path}")

        if path is not None and os.path.exists(path):
            print(f"Returning file response: {path}-{os.path.splitext(path)}")
            return FileResponse(path)

        return {"exception": f"Unable to find or compile checkpoint."}

    @app.get("/dreambooth/list_checkpoints")
    async def get_checkpoints(
            api_key: str = Query("", description="If an API key is set, this must be present.", )
    ):
        """
        Collect the current list of available source checkpoints.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        sd_models.list_models()
        ckpt_list = sd_models.checkpoints_list
        return JSONResponse(content=ckpt_list)

    @app.get("/dreambooth/samples")
    async def generate_samples(
            model_name: str = Query(description="The model name to use for generating samples."),
            sample_prompt: str = Query(description="The prompt to use to generate sample images."),
            num_images: int = Query(1, description="The number of sample images to generate."),
            batch_size: int = Query(1, description="How many images to generate at once."),
            lora_model_path: str = Query("", description="The path to a lora model to use when generating images."),
            lora_rank: int = Query(1,
                                       description="The rank of LoRA models (the amount of data to retain in the LoRA file after training)"),
            lora_weight: float = Query(1.0,
                                       description="The weight of the lora unet when merging with the base model."),
            lora_txt_weight: float = Query(1.0,
                                           description="The weight of the lora text encoder when merging with the base model"),
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
        if db_shared.status.job_count != 0:
            print("Something is already running.")
            return JSONResponse(content={"message": "Job already in progress.", "status": db_shared.status.dict()})
        db_shared.status.begin()
        images, msg, status = ui_samples(
            model_dir=model_name,
            save_sample_prompt=sample_prompt,
            num_samples=num_images,
            sample_batch_size=batch_size,
            lora_model_path=lora_model_path,
            lora_rank=lora_rank,
            lora_weight=lora_weight,
            lora_txt_weight=lora_txt_weight,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            scale=scale
        )
        db_shared.status.end()
        if len(images) > 1:
            return zip_files(model_name, images, "_sample")
        else:
            img_byte_arr = io.BytesIO()
            file = images[0]
            file.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")

    @app.post("/dreambooth/classifiers")
    async def generate_classes(
            model_name: str = Query(description="The model name to generate classifiers for."),
            use_txt2img: bool = Query("", description="Use Txt2Image to generate classifiers."),
            api_key: str = Query("", description="If an API key is set, this must be present.")
    ):
        """
        Generate classification images for a model based on a saved config.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        if model_name is None or model_name == "":
            return JSONResponse(status_code=422, content={"message": "Invalid model name."})
        config = from_file(model_name)
        if config is None:
            return JSONResponse(status_code=422, content={"message": "Invalid config."})
        if db_shared.status.job_count != 0:
            print("Something is already running.")
            return JSONResponse(content={"message": "Job already in progress.", "status": db_shared.status.dict()})
        db_shared.status.begin()
        run_in_background(
            generate_classifiers,
            config,
            use_txt2img
        )
        return JSONResponse(content={"message":"Generating classifiers..."})


    @app.get("/dreambooth/classifiers")
    async def get_classifiers(
            model_name: str = Query(description="The model name to retrieve classifiers for."),
            concept_idx: int = Query(-1,
                                     description="If set, will retrieve the specified concept's class images. Otherwise, all class images will be retrieved."),
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
            if class_images_dir == "" or class_images_dir is None or class_images_dir == db_shared.script_path:
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

        Request body should be a JSON Object. Primary key is 'imageList'.

        'imageList' is a list of objects. Each object should have three values:
        'data' - A base64-encoded string containing the binary data of the image.
        'name' - The filename to store the image under.
        'txt' - The caption for the image. Will be stored in a text file beside the image.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check

        root_img_path = os.path.join(db_shared.script_path, "..", "InstanceImages")
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
        return JSONResponse(content=inst_datas)


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(dreambooth_api)
    print("SD-Webui API layer loaded")
except:
    pass
