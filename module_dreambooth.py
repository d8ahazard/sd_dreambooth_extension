import asyncio
import logging
import os
import shutil

from core.handlers.models import ModelHandler
from core.handlers.status import StatusHandler
from core.handlers.websocket import SocketHandler
from core.modules.base.module_base import BaseModule
from fastapi import FastAPI

import scripts.api
from dreambooth.dataclasses.db_config import DreamboothConfig
from dreambooth.sd_to_diff import extract_checkpoint
from module_src.gradio_parser import parse_gr_code

logger = logging.getLogger(__name__)


class DreamboothModule(BaseModule):

    def __init__(self):
        self.name: str = "Dreambooth"
        self.path = os.path.abspath(os.path.dirname(__file__))
        self.model_handler = ModelHandler()
        super().__init__(self.name, self.path)

    def initialize(self, app: FastAPI, handler: SocketHandler):
        self._initialize_api(app)
        self._initialize_websocket(handler)

    def _initialize_api(self, app: FastAPI):
        return scripts.api.dreambooth_api(None, app)

    def _initialize_websocket(self, handler: SocketHandler):
        handler.register("train_dreambooth", _train_dreambooth)
        handler.register("create_dreambooth", _create_model)
        handler.register("get_layout", _get_layout)


async def _train_dreambooth(data):
    logger.debug(f"Train dreambooth called: {data}")
    await asyncio.sleep(1)
    return {"status": "Training started."}


async def _create_model(data):
    mh = ModelHandler(user_name=data["user"] if "user" in data else None)
    sh = StatusHandler(user_name=data["user"] if "user" in data else None)
    msg_id = data["id"]
    logger.debug(f"Full message: {data}")
    data = data["data"] if "data" in data else None
    logger.debug(f"Create model called: {data}")
    model_name = data["new_model_name"] if "new_model_name" in data else None
    src = data["new_model_src"]["path"]
    shared_src = data["new_model_shared_src"]["path"] if "new_model_shared_src" in data else None
    from_hub = data["create_from_hub"] if "create_from_hub" in data else False
    logger.debug(f"SRC - {src} and {from_hub}")
    if src and not from_hub:
        sh.start(1, "Copying source weights.")
        copy_model(model_name, src, data["512_model"], mh)
        sh.step()
        sh.end("Model created.")
    else:
        loop = asyncio.get_running_loop()
        loop.create_task(extract_checkpoint(
            model_name,
            src,
            shared_src,
            True,
            data["new_model_url"],
            data["new_model_token"],
            data["new_model_extract_ema"],
            data["train_unfrozen"],
            data["512_model"]
        ))
    return {"name": "create_model", "message": "Creating model.", "id": msg_id}


def copy_model(model_name: str, src: str, is_512: bool, mh: ModelHandler):
    logger.debug("Copying model!")
    models_path = mh.models_path
    logger.debug(f"Models paths: {models_path}")
    model_dir = models_path[0]
    dreambooth_models_path = os.path.join(model_dir, "dreambooth")
    dest_dir = os.path.join(model_dir, "dreambooth", model_name, "working")
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir, True)
    if not os.path.exists(dest_dir):
        shutil.copytree(src, dest_dir)
        cfg = DreamboothConfig(model_name=model_name, src=src, resolution=is_512, models_path=dreambooth_models_path)
        cfg.save()
    else:
        logger.debug(f"Destination directory '{dest_dir}' already exists, skipping copy.")


async def _get_layout(data):
    logger.debug(f"Get layout called: {data}")
    layout_file = os.path.join(os.path.dirname(__file__), "scripts", "main.py")
    logger.debug(f"Trying to parse: {layout_file}")
    output = parse_gr_code(layout_file)
    logger.debug(f"Output: {output}")
    return {"status": "Layout created.", "layout": output}