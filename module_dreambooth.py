import asyncio
import gc
import json
import logging
import os
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Dict

import torch
from core.handlers.config import ConfigHandler
from core.handlers.models import ModelHandler, ModelManager
from core.handlers.status import StatusHandler
from core.handlers.websocket import SocketHandler
from core.modules.base.module_base import BaseModule
from fastapi import FastAPI

import scripts.api
from dreambooth import shared
from dreambooth.dataclasses.db_config import DreamboothConfig, from_file
from dreambooth.sd_to_diff import extract_checkpoint
from dreambooth.train_dreambooth import main
from module_src.gradio_parser import parse_gr_code

logger = logging.getLogger(__name__)


class DreamboothModule(BaseModule):

    def __init__(self):
        self.id = "dreambooth"
        self.name: str = "Dreambooth"
        self.path = os.path.abspath(os.path.dirname(__file__))
        self.model_handler = ModelHandler()
        super().__init__(self.id, self.name, self.path)

    def initialize(self, app: FastAPI, handler: SocketHandler):
        self._initialize_api(app)
        self._initialize_websocket(handler)
        defaults_base_file = os.path.join(os.path.dirname(__file__), "templates", "db_config.json")
        if os.path.exists(defaults_base_file):
            ch = ConfigHandler()
            ch.set_config_protected(json.load(open(defaults_base_file, "r")), "dreambooth_model_defaults")

    def _initialize_api(self, app: FastAPI):
        return scripts.api.dreambooth_api(None, app)

    def _initialize_websocket(self, handler: SocketHandler):
        handler.register("train_dreambooth", _train_dreambooth)
        handler.register("create_dreambooth", _create_model)
        handler.register("get_db_config", _get_model_config)
        handler.register("save_db_config", _set_model_config)
        handler.register("get_layout", _get_layout)
        handler.register("get_db_vars", _get_db_vars)


async def _get_db_vars(request):
    from dreambooth.utils.utils import (
        list_attention,
        list_precisions,
        list_optimizer,
        list_schedulers,
    )
    from dreambooth.utils.image_utils import get_scheduler_names

    attentions = list_attention()
    precisions = list_precisions()
    optimizers = list_optimizer()
    schedulers = list_schedulers()
    infer_schedulers = get_scheduler_names()
    return {
        "attentions": attentions,
        "precisions": precisions,
        "optimizers": optimizers,
        "schedulers": schedulers,
        "infer_schedulers": infer_schedulers
    }


async def _train_dreambooth(request):
    user = request["user"] if "user" in request else None
    config = await _set_model_config(request, True)
    mh = ModelHandler(user_name=user)
    mm = ModelManager()
    sh = StatusHandler(user_name=user, target="dreamProgress")
    mm.to_cpu()
    shared.db_model_config = config
    try:
        torch.cuda.empty_cache()
        gc.collect()
    except:
        pass
    sh.start(0, "Starting Dreambooth Training...")
    await sh.send_async()
    result = {"message": "Training complete."}
    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, lambda: (
                sh.start(0, "Starting Dreambooth Training..."),
                main(user=user)
            ))
    except Exception as e:
        logger.error(f"Error in training: {e}")
        traceback.print_exc()
        result = {"message": f"Error in training: {e}"}
    try:
        gc.collect()
        torch.cuda.empty_cache()
    except:
        pass
    sh.end(result["message"])
    return result


async def _create_model(data):
    target = data["target"] if "target" in data else None
    mh = ModelHandler(user_name=data["user"] if "user" in data else None)
    sh = StatusHandler(user_name=data["user"] if "user" in data else None, target=target)
    logger.debug(f"Full message: {data}")
    data = data["data"] if "data" in data else None
    logger.debug(f"Create model called: {data}")
    model_name = data["new_model_name"] if "new_model_name" in data else None
    src_hash = data["new_model_src"]
    src_model = await mh.find_model("diffusers", src_hash)
    logger.debug(f"SRC Model result: {src_model}")
    src = src_model.path if src_model else None
    shared_src = data["new_model_shared_src"] if "new_model_shared_src" in data else None
    from_hub = data["create_from_hub"] if "create_from_hub" in data else False
    logger.debug(f"SRC - {src} and {from_hub}")
    if not src:
        logger.debug("Unable to find source model.")
        return {"status": "Unable to find source model.."}
    sh.start(desc=f"Creating model: {model_name}")
    if src and not from_hub:
        sh.update("status", "Copying model.")
        await sh.send_async()
        dest = await copy_model(model_name, src, data["512_model"], mh, sh)
        mh.refresh("dreambooth", dest, model_name)
    else:
        sh.update("status", "Extracting model.")
        await sh.send_async()
        extract_checkpoint(new_model_name=model_name,
                           checkpoint_file=src,
                           extract_ema=False,
                           train_unfrozen=data["train_unfrozen"],
                           is_512=data["512_model"]
                           )
        mh.refresh("dreambooth")
    sh.end(f"Created model: {model_name}")
    return {"status": "Model created."}


async def copy_model(model_name: str, src: str, is_512: bool, mh: ModelHandler, sh: StatusHandler):
    models_path = mh.models_path
    logger.debug(f"Models paths: {models_path}")
    model_dir = models_path[0]
    dreambooth_models_path = os.path.join(model_dir, "dreambooth")
    dest_dir = os.path.join(model_dir, "dreambooth", model_name, "working")
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir, True)
    ch = ConfigHandler(user_name=mh.user_name)
    base = ch.get_module_defaults("dreambooth_model_defaults")
    user_base = ch.get_config_user("dreambooth_model_defaults")
    logger.debug(f"User base: {user_base}")
    if base is not None:
        if user_base is not None:
            base = {**base, **user_base}
        else:
            ch.set_config_user(base, "dreambooth_model_defaults")
    else:
        logger.debug("Unable to find base model config: dreambooth_model_defaults")
    if not os.path.exists(dest_dir):
        logger.debug(f"Copying model from {src} to {dest_dir}")
        await copy_directory(src, dest_dir, sh)
        cfg = DreamboothConfig(model_name=model_name, src=src, resolution=512 if is_512 else 768, models_path=dreambooth_models_path)
        cfg.load_params(base)
        cfg.save()

    else:
        logger.debug(f"Destination directory '{dest_dir}' already exists, skipping copy.")
    logger.debug("Model copied.")
    return dest_dir


async def copy_directory(src_dir, dest_dir, sh: StatusHandler):
    total_size = get_directory_size(src_dir)
    sh.start(100, "Copying source weights.")
    copied_pct = 0
    copied_size = 0
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_path = os.path.join(root, file)
            # Get the name of the parent of the file
            parent = os.path.basename(os.path.dirname(src_path))
            sh.update(items={"status_2": f"Copying {parent}{os.sep}{file}"})
            await sh.send_async()

            dest_path = os.path.join(dest_dir, os.path.relpath(src_path, src_dir))
            dest_dirname = os.path.dirname(dest_path)
            if not os.path.exists(dest_dirname):
                logger.debug("Making directory(md): " + dest_dirname)
                os.makedirs(dest_dirname)
            shutil.copy2(src_path, dest_path)
            copied_size += os.path.getsize(src_path)
            current_pct = int(copied_size / total_size * 100)
            if current_pct > copied_pct:
                sh.update(items={"progress_1_current": current_pct})
                await sh.send_async()
                copied_pct = current_pct


def get_directory_size(dir_path):
    total_size = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    return total_size


async def _get_layout(data):
    logger.debug(f"Get layout called: {data}")
    layout_file = os.path.join(os.path.dirname(__file__), "scripts", "main.py")
    logger.debug(f"Trying to parse: {layout_file}")
    output = parse_gr_code(layout_file)
    logger.debug(f"Output: {output}")
    return {"status": "Layout created.", "layout": output}


async def _get_model_config(data, return_json=True):
    logger.debug(f"Get model called: {data}")
    model = data["data"]["model"]
    config = from_file(model["name"], os.path.dirname(model["path"]))
    if config.concepts_path and os.path.exists(config.concepts_path):
        with open(config.concepts_path, "r") as f:
            config.concepts_list = json.load(f)
        config.concepts_path = ""
        config.use_concepts = False
        config.save()
    if return_json:
        return {"config": config.__dict__}
    return config


async def _set_model_config(data: dict, return_config: bool = False) -> Union[Dict, DreamboothConfig]:
    logger.debug(f"Set model called: {data}")
    model = data["data"]["model"]
    training_params = data["data"]
    del training_params["model"]
    config = from_file(model["name"], os.path.dirname(model["path"]))
    config.load_params(training_params)
    config.save()
    return {"config": config.__dict__} if not return_config else config
