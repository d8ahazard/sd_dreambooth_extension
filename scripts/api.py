import base64
import functools
import hashlib
import io
import json
import logging
import os
import shutil
import traceback
import zipfile
from pathlib import Path
from typing import List, Union

import requests
from PIL import Image
from fastapi import FastAPI, Response, Query, Body, Form, Header
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from starlette import status
from starlette.requests import Request

try:
    from dreambooth import shared
    from dreambooth.dataclasses.db_concept import Concept
    from dreambooth.dataclasses.db_config import from_file, DreamboothConfig
    from dreambooth.diff_to_sd import compile_checkpoint
    from dreambooth.secret import get_secret
    from dreambooth.shared import DreamState
    from dreambooth.ui_functions import create_model, generate_samples, \
        start_training
    from dreambooth.utils.gen_utils import generate_classifiers
    from dreambooth.utils.image_utils import get_images
    from dreambooth.utils.model_utils import get_db_models, get_lora_models
except:
    print("Exception importing api")
    traceback.print_exc()

if os.environ.get("DEBUG_API", False):
    logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


class InstanceData(BaseModel):
    data: str = Field(title="File data", description="Base64 representation of the file or URL")
    name: str = Field(title="File name", description="File name to save image as")
    txt: str = Field(title="Prompt", description="Training prompt for image")


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


import asyncio

active = False


def is_running():
    return False


def run_in_background(func, *args, **kwargs):
    """
    Wrapper function to run a non-asynchronous method as a task in the event loop.
    """

    async def wrapper():
        global active
        new_func = functools.partial(func, *args, **kwargs)
        await asyncio.get_running_loop().run_in_executor(None, new_func)
        active = False

    asyncio.create_task(wrapper())


def zip_files(db_model_name, files, name_part=""):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a",
                         zipfile.ZIP_DEFLATED, False) as zip_file:
        for file in files:
            if isinstance(file, str):
                logger.debug(f"Zipping img: {file}")
                if os.path.exists(file) and os.path.isfile(file):
                    parent_path = os.path.join(Path(file).parent, Path(file).name)
                    zip_file.write(file, arcname=parent_path)
                    check_txt = os.path.join(os.path.splitext(file)[0], ".txt")
                    if os.path.exists(check_txt):
                        logger.debug(f"Zipping txt: {check_txt}")
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
    if current_key is not None and current_key:
        if not key:
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


def dreambooth_api(_, app: FastAPI):
    logger.debug("Loading Dreambooth API Endpoints.")
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
        )

    @app.get("/dreambooth/cancel")
    async def cancel_jobs(
            api_key: str = Query("", description="If an API key is set, this must be present.", )) -> \
            JSONResponse:
        """
        Check the current state of Dreambooth processes.
        @return:
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        if shared.status.job_count == 0:
            return JSONResponse(content={"message": "Nothing to cancel."})
        shared.status.interrupted = True
        return JSONResponse(content={"message": f"Processes cancelled."})

    @app.get("/dreambooth/checkpoint")
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
        if not model_name:
            return JSONResponse(status_code=422, content={"message": "Invalid model name."})
        config = from_file(model_name)
        if config is None:
            return JSONResponse(status_code=422, content={"message": "Invalid config."})

        status = is_running()
        if status:
            return status

        path = None
        save_model_name = config.model_name
        if config.custom_model_name:
            save_model_name = config.custom_model_name
        if skip_build:
            ckpt_dir = shared.ckpt_dir
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
            if config.save_safetensors:
                checkpoint_path = checkpoint_path.replace(".ckpt", ".safetensors")
            logger.debug(f"Looking for checkpoint at {checkpoint_path}")
            if os.path.exists(checkpoint_path):
                logger.debug("Existing checkpoint found, returning.")
                path = checkpoint_path
            else:
                skip_build = False
        if not skip_build:
            global active
            shared.status.begin()
            active = True
            ckpt_result = compile_checkpoint(model_name, reload_models=False, log=False)
            active = False
            if "Checkpoint compiled successfully" in ckpt_result:
                path = ckpt_result.replace("Checkpoint compiled successfully:", "").strip()
                logger.debug(f"Checkpoint aved to path: {path}")

        if path is not None and os.path.exists(path):
            logger.debug(f"Returning file response: {path}-{os.path.splitext(path)}")
            return FileResponse(path)

        return {"exception": f"Unable to find or compile checkpoint."}

    @app.get("/dreambooth/checkpoints")
    async def get_checkpoints(
            api_key: str = Query("", description="If an API key is set, this must be present.", )
    ):
        """
        Collect the current list of available source checkpoints.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check

        ckpt_list = {}
        try:
            from modules import sd_models

            sd_models.list_models()
            ckpt_list = sd_models.checkpoints_list
        except:
            pass
        models = []
        for key, _ in ckpt_list.items():
            models.append(key)
        return JSONResponse(content=models)

    @app.post("/dreambooth/classifiers")
    async def generate_classes(
            model_name: str = Form(description="The model name to generate classifiers for."),
            class_gen_method: str = Form("Native Diffusers", description="Image Generation Library."),
            api_key: str = Form("", description="If an API key is set, this must be present.")
    ):
        """
        Generate classification images for a model based on a saved config.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        if not model_name:
            return JSONResponse(status_code=422, content={"message": "Invalid model name."})
        config = from_file(model_name)
        if config is None:
            return JSONResponse(status_code=422, content={"message": "Invalid config."})

        status = is_running()
        if status:
            return status
        global active
        active = True
        shared.status.begin()
        run_in_background(
            generate_classifiers,
            config,
            class_gen_method
        )
        active = False
        return JSONResponse(content={"message": "Generating classifiers..."})

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
        concepts = config.concepts()
        concept_dict = {}
        out_images = []
        if concept_idx >= 0:
            if len(concepts) - 1 >= concept_idx:
                logger.debug(f"Returning class images for concept {concept_idx}")
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
                logger.debug(f"Class image dir is not set, defaulting to {class_images_dir}")
            if os.path.exists(class_images_dir):
                class_images = get_images(class_images_dir)
                for image in class_images:
                    out_images.append(str(image))

        if len(out_images) > 0:
            return zip_files(model_name, out_images, "_class")
        else:
            return {"Result": "No images found."}

    @app.post("/dreambooth/concept")
    async def set_model_concept(
            model_name: str = Form(description="The model name to fetch config for."),
            instance_dir: str = Form("", description="The directory containing training images."),
            instance_token: str = Form("", description="The instance token to use."),
            class_token: str = Form("", description="The class token to use."),
            api_key: str = Form("", description="If an API key is set, this must be present."),
            concept: Union[Concept, None] = Body(None, description="A concept to update or add to the model.")
    ) -> JSONResponse:
        """
        Add or update a concept. Provide either a full json concept or path to instance dir.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        if not model_name:
            return JSONResponse(status_code=422, content={"message": "Invalid model name."})
        config = from_file(model_name)
        if config is None:
            return JSONResponse(status_code=422, content={"message": "Invalid config."})
        new_concepts = []
        if concept is None and instance_dir:
            new_concept = Concept()
            new_concept.instance_data_dir = instance_dir
            new_concept.instance_token = instance_token
            new_concept.class_token = class_token
            new_concept.class_prompt = "[filewords]"
            new_concept.instance_prompt = "[filewords]"
            new_concept.save_sample_prompt = "[filewords]"
            new_concept.is_valid = os.path.exists(instance_dir)

        existing_concepts = config.concepts()
        replaced = False
        for ex_concept in existing_concepts:
            if ex_concept.instance_data_dir == concept.instance_data_dir:
                new_concepts.append(concept.__dict__)
                replaced = True
            else:
                new_concepts.append(ex_concept)
        if not replaced:
            new_concepts.append(concept.__dict__)
        config.concepts_list = new_concepts
        config.save()
        return JSONResponse(content=config.concepts())

    @app.get("/dreambooth/concepts")
    async def get_model_concepts(
            model_name: str = Query(description="The model name to fetch config for."),
            api_key: str = Query("", description="If an API key is set, this must be present.", )
    ) -> JSONResponse:
        """
        Get a model's concepts.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        if not model_name:
            return JSONResponse(status_code=422, content={"message": "Invalid model name."})
        config = from_file(model_name)
        if config is None:
            return JSONResponse(status_code=422, content={"message": "Invalid config."})

        return JSONResponse(content=config.concepts())

    @app.post("/dreambooth/concepts")
    async def set_model_concepts(
            model_name: str = Form(description="The model name to fetch config for."),
            api_key: str = Form("", description="If an API key is set, this must be present."),
            concepts: List[Concept] = Body()
    ) -> JSONResponse:
        """
        Replace a full concepts list.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        if not model_name:
            return JSONResponse(status_code=422, content={"message": "Invalid model name."})
        config = from_file(model_name)
        if config is None:
            return JSONResponse(status_code=422, content={"message": "Invalid config."})
        new_concepts = []
        for concept in concepts:
            new_concepts.append(concept.__dict__)
        config.concepts_list = new_concepts
        config.save()
        return JSONResponse(content=config.concepts())

    @app.post("/dreambooth/createModel")
    async def create_db_model(
            new_model_name: str = Query(description="The name of the model to create.", ),
            new_model_src: str = Query(description="The source checkpoint to extract to create this model.", ),
            new_model_scheduler: str = Query("ddim", description="The scheduler to use. V2+ models ignore this.", ),
            create_from_hub: bool = Query(False, description="Create this model from the hub", ),
            new_model_url: str = Query(None,
                                       description="The hub URL to use for this model. Must contain diffusers model.", ),
            is_512: bool = Query(False,
                                 description="Whether or not the model is 512x resolution.", ),
            train_unfrozen: bool = Query(True,
                                         description="Un-freeze the model.", ),
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

        if not new_model_name:
            return JSONResponse(status_code=422, content={"message": "Invalid model name."})

        status = is_running()
        if status:
            return status

        logger.debug("Creating new Checkpoint: " + new_model_name)
        res = create_model(new_model_name,
                           new_model_src,
                           create_from_hub,
                           new_model_url,
                           new_model_token,
                           new_model_extract_ema,
                           train_unfrozen,
                           is_512)

        return JSONResponse(res[-1])

    @app.delete("/dreambooth/model")
    async def delete_model(
            model_name: str = Form(description="The model to delete."),
            api_key: str = Form("", description="If an API key is set, this must be present."),
    ) -> JSONResponse:
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        if not model_name:
            return JSONResponse(status_code=422, content={"message": "Invalid model name."})
        config = from_file(model_name)
        if config is None:
            return JSONResponse(status_code=422, content={"message": "Invalid config."})
        model_dir = config.model_dir
        models_path = os.path.join(shared.models_path, "stable-diffusion")
        model_base = config.custom_model_name if config.custom_model_name else config.model_name
        if config.use_subdir:
            models_path = os.path.join(models_path, model_base)

        model_files = os.listdir(models_path)
        for mf in model_files:
            rev = mf.split("_")[-1]
            try:
                revision = int(rev)
                if mf == f"{model_base}_{revision}.safetensors":
                    full_file = os.path.join(models_path, mf)
                    print(f"Removing model: {full_file}")
                    os.remove(full_file)
            except:
                pass
        try:
            shutil.rmtree(model_dir,True)
        except:
            pass

        return JSONResponse(f"Model {model_name} has been deleted.")

    @app.get("/dreambooth/model_config")
    async def get_model_config(
            model_name: str = Query(description="The model name to fetch config for."),
            api_key: str = Query("", description="If an API key is set, this must be present.", )
    ) -> JSONResponse:
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

        return JSONResponse(content=config.__dict__)

    @app.post("/dreambooth/model_config")
    async def set_model_config(
            model_cfg: DreamboothConfig = Body(description="The config to save"),
            api_key: str = Header(description="If an API key is set, this must be present.", default="")
    ):
        """
        Save a model config from JSON.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        try:
            logger.debug("Create config")
            config = DreamboothConfig()
            for key in model_cfg.dict():
                if key in config.__dict__:
                    config.__dict__[key] = model_cfg.dict()[key]
            config.save()
            logger.debug("Saved?")
            return JSONResponse(content=config.__dict__)
        except Exception as e:
            traceback.print_exc()
            return {"Exception saving model": f"{e}"}

    @app.post("/dreambooth/model_params")
    async def set_model_params(
            model_name: str = Query(description="The model name to update params for."),
            api_key: str = Query("", description="If an API key is set, this must be present."),
            params: str = Query(description="A json string representing a dictionary of parameters to set.")
    ) -> JSONResponse:
        """
        Update an existing model configuration's parameters from a dictionary of values.
        """
        params = json.loads(params)
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        if model_name is None or model_name == "":
            return JSONResponse(status_code=422, content={"message": "Invalid model name."})
        config = from_file(model_name)
        if config is None:
            return JSONResponse(status_code=422, content={"message": "Invalid config."})
        logger.debug(f"Loading new params: {params}")
        config.load_params(params)
        config.save()
        return JSONResponse(content=config.__dict__)

    @app.get("/dreambooth/models")
    async def get_models(
            api_key: str = Query("", description="If an API key is set, this must be present."),
    ) -> JSONResponse:
        """

        Args:
            api_key: The api key

        Returns: A list of Dreambooth model names.

        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        models = get_db_models()
        return JSONResponse(models)

    @app.get("/dreambooth/models_lora")
    async def get_models_lora(
            api_key: str = Query("", description="If an API key is set, this must be present."),
            model_name: str = Query(description="The model name to query for lora files."),
    ) -> JSONResponse:
        """

        Args:
            api_key: API Key.
            model_name: The model name to query for lora files.

        Returns: A list of LoRA Models.

        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check

        config = from_file(model_name)
        if model_name and config is None:
            return JSONResponse("Config not found")

        models = get_lora_models(config)
        return JSONResponse(models)

    @app.get("/dreambooth/samples")
    async def api_generate_samples(
            model_name: str = Query(description="The model name to use for generating samples."),
            sample_prompt: str = Query("", description="The prompt to use to generate sample images."),
            sample_prompt_file: str = Query("", description="A file of pre-generated sample prompts."),
            negative_prompt: str = Query("", description="An optional negative prompt to use when generating images."),
            width: int = Query(512, description="Sample width"),
            height: int = Query(512, description="Sample height"),
            num_images: int = Query(1, description="The number of sample images to generate."),
            batch_size: int = Query(1, description="How many images to generate at once."),
            seed: int = Query(-1, description="The seed to use when generating samples"),
            steps: int = Query(60, description="Number of sampling steps to use when generating images."),
            scale: float = Query(7.5, description="CFG scale to use when generating images."),
            class_gen_method: str = Query("Native Diffusers", description="Image Generation Library."),
            scheduler: str = Query("DEISMultistep", description="Sampler to use if not using txt2img"),
            api_key: str = Query("", description="If an API key is set, this must be present.", )
    ):
        """
        Generate sample images for a specified model.
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check

        status = is_running()
        if status:
            return status

        shared.status.begin()
        config = from_file(model_name)
        if config is None:
            return JSONResponse("Config not found")

        images, prompts, status = generate_samples(
            model_name,
            prompt=sample_prompt,
            prompt_file=sample_prompt_file,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_samples=num_images,
            batch_size=batch_size,
            seed=seed,
            scale=scale,
            steps=steps,
            class_gen_method=class_gen_method,
            scheduler=scheduler
        )

        if len(images) > 1:
            return zip_files(model_name, images, "_sample")
        else:
            file = images[0]
            image = Image.open(file)
            img_byte_arr = image.tobytes()

        return Response(content=img_byte_arr, media_type="image/png")

    @app.get("/dreambooth/status")
    async def check_status(
            api_key: str = Query("", description="If an API key is set, this must be present.", )) -> \
            JSONResponse:
        """
        Check the current state of Dreambooth processes.
        @return:
        """
        key_check = check_api_key(api_key)
        if key_check is not None:
            return key_check
        return JSONResponse(content={"current_state": f"{json.dumps(shared.status.dict())}"})

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
        shared.status.set_current_image()
        images = shared.status.current_image
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

    @app.post("/dreambooth/start_training")
    async def train(
            model_name: str = Query(None,
                                    description="The model name to load params for.", ),
            use_tx2img: bool = Query(True, description="Use txt2img to generate class images."),
            api_key: str = Query("", description="If an API key is set, this must be present.")
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

        status = is_running()
        if status:
            return status

        logger.debug("Starting Training")
        shared.status.begin()
        run_in_background(start_training, model_name, use_tx2img)
        return {"Status": "Training started."}

    @app.post("/dreambooth/upload")
    async def upload_db_images(
            model_name: str = Query(description="The model name to upload images for."),
            instance_name: str = Query(description="The concept/instance name the images are for."),
            create_concept: bool = Query(True,
                                         description="Enable to automatically append the new concept to the model config."),
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
        logger.debug("API UPLOAD STARTED.")
        key_check = check_api_key(api_key)
        if key_check is not None:
            logger.debug("NO KEY")
            return key_check

        root_img_path = os.path.join(shared.script_path, "..", "InstanceImages")
        if not os.path.exists(root_img_path):
            logger.debug(f"Creating root instance dir: {root_img_path}")
            os.makedirs(root_img_path)
        else:
            logger.debug(f"Root dir exists already: {root_img_path}")

        image_dir = os.path.join(root_img_path, model_name, instance_name)
        image_dir = os.path.abspath(image_dir)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            logger.debug(f"Input data: {images}")

        image_paths = []
        for img_data in images.imageList:
            img = base64_to_pil(img_data.data)
            name = img_data.name
            prompt = img_data.txt
            logger.debug(f"Input prompt for image: {prompt} {name}")
            image_path = os.path.join(image_dir, name)
            text_path = os.path.splitext(image_path)[0]
            text_path = F"{text_path}.txt"
            logger.debug(f"Saving image to: {image_path}")
            img.save(image_path)
            logger.debug(f"Saving prompt ({prompt}) to: {text_path}")
            with open(text_path, "w") as tx_file:
                tx_file.writelines(prompt)
                logger.debug(f"Saved prompt text to: {text_path}")
            image_paths.append(image_path)

        status = {"Status": f"Saved {len(image_paths)} images.", "Image dir": {image_dir}}
        if create_concept:
            config = from_file(model_name)
            logger.debug(f"Creating concept: {model_name}")
            if config is None:
                status["Status"] += " Unable to load model config."
            new_concept = Concept()
            new_concept.instance_data_dir = image_dir
            new_concept.class_prompt = "[filewords]"
            new_concept.instance_prompt = "[filewords]"
            new_concept.save_sample_prompt = "[filewords]"
            new_concept.is_valid = True
            logger.debug(f"New concept: {new_concept}")
            new_concepts = []
            replaced = False
            for concept in config.concepts():
                if concept.instance_data_dir == new_concept.instance_data_dir:
                    new_concepts.append(new_concept.__dict__)
                    replaced = True
                else:
                    new_concepts.append(concept.__dict__)
            if not replaced:
                new_concepts.append(new_concept.__dict__)
            config.concepts_list = new_concepts
            config.save()
            logger.debug("Saved concepts.")
            status["Concepts"] = config.concepts_list

        return status

    @app.post("/dreambooth/upload_url")
    async def upload_db_images_url(
            model_name: str = Query(description="The model name to upload images for."),
            instance_name: str = Query(description="The concept/instance name the images are for."),
            create_concept: bool = Query(True,
                                         description="Enable to automatically append the new concept to the model config."),
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
        logger.debug("API UPLOAD STARTED.")
        key_check = check_api_key(api_key)
        if key_check is not None:
            logger.debug("NO KEY")
            return key_check

        root_img_path = os.path.join(shared.script_path, "..", "InstanceImages")
        if not os.path.exists(root_img_path):
            logger.debug(f"Creating root instance dir: {root_img_path}")
            os.makedirs(root_img_path)
        else:
            logger.debug(f"Root dir exists already: {root_img_path}")

        image_dir = os.path.join(root_img_path, model_name, instance_name)
        image_dir = os.path.abspath(image_dir)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            logger.debug(f"Input data: {images}")

        image_paths = []
        for img_data in images.imageList:
            img_url = img_data.data
            name = img_data.name
            prompt = img_data.txt
            logger.debug(f"Input prompt for image: {prompt} {name}")
            image_path = os.path.join(image_dir, name)
            text_path = os.path.splitext(image_path)[0]
            text_path = F"{text_path}.txt"
            logger.debug(f"Saving image to: {image_path}")
            try:
                response = requests.get(img_url, stream=True)
                response.raise_for_status()
                with open(image_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                logger.exception(f"Error downloading image from {img_url}: {e}")
                continue
            logger.debug(f"Saving prompt ({prompt}) to: {text_path}")
            with open(text_path, "w") as tx_file:
                tx_file.writelines(prompt)
                logger.debug(f"Saved prompt text to: {text_path}")
            image_paths.append(image_path)

        status = {"Status": f"Saved {len(image_paths)} images.", "Image dir": {image_dir}}
        if create_concept:
            config = from_file(model_name)
            logger.debug(f"Creating concept: {model_name}")
            if config is None:
                status["Status"] += " Unable to load model config."
            new_concept = Concept()
            new_concept.instance_data_dir = image_dir
            new_concept.class_prompt = "[filewords]"
            new_concept.instance_prompt = "[filewords]"
            new_concept.save_sample_prompt = "[filewords]"
            new_concept.is_valid = True
            logger.debug(f"New concept: {new_concept}")
            new_concepts = []
            replaced = False
            for concept in config.concepts():
                if concept.instance_data_dir == new_concept.instance_data_dir:
                    new_concepts.append(new_concept.__dict__)
                    replaced = True
                else:
                    new_concepts.append(concept.__dict__)
            if not replaced:
                new_concepts.append(new_concept.__dict__)
            config.concepts_list = new_concepts
            config.save()
            logger.debug("Saved concepts.")
            status["Concepts"] = config.concepts_list

        return status


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(dreambooth_api)
    logger.debug("SD-Webui API layer loaded")
except:
    logger.debug("Unable to import script callbacks.")
    pass
