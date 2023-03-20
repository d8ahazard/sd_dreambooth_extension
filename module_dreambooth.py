import asyncio
import logging
import os
import shutil

from core.handlers.models import ModelHandler
from core.handlers.websocket import SocketHandler
from core.modules.base.module_base import BaseModule
from fastapi import FastAPI

import scripts.api
from dreambooth.dataclasses.db_config import DreamboothConfig
from dreambooth.sd_to_diff import extract_checkpoint

logger = logging.getLogger(__name__)

class DreamboothModule(BaseModule):

    def __init__(self):
        self.name: str = "Dreambooth"
        self.path = os.path.abspath(os.path.dirname(__file__))
        self.model_handler = ModelHandler()
        super().__init__(self.name, self.path)
        socket_handler = SocketHandler()
        socket_handler.register("train_dreambooth", self._train_dreambooth)
        socket_handler.register("create_dreambooth", self._create_model)

    async def _train_dreambooth(self, data):
        self.logger.debug(f"Train dreambooth called: {data}")

    async def _create_model(self, data):
        msg_id = data["id"]
        data = data["data"] if "data" in data else None
        self.logger.debug(f"Create model called: {data}")
        model_name = data["new_model_name"] if "new_model_name" in data else None
        src = data["new_model_src"]["path"]
        from_hub = data["create_from_hub"] if "create_from_hub" in data else False
        self.logger.debug(f"SRC - {src} and {from_hub}")
        if src and not from_hub:
            self.copy_model(model_name, src, data["512_model"])
        else:
            loop = asyncio.get_running_loop()
            loop.create_task(extract_checkpoint(
                model_name,
                src,
                True,
                data["new_model_url"],
                data["new_model_token"],
                data["new_model_extract_ema"],
                data["train_unfrozen"],
                data["512_model"]
            ))
        return {"name": "create_model", "message": "Creating model.", "id": msg_id}

    def initialize(self, app: FastAPI, handler: SocketHandler):
        self._initialize_api(app)

    def _initialize_api(self, app: FastAPI):
        return scripts.api.dreambooth_api(None, app)

    def copy_model(self, model_name: str, src: str, is_512: bool):
        logger.debug("Copying model!")
        model_dir = self.model_handler.models_path
        dest_dir = os.path.join(model_dir, "dreambooth", model_name, "working")
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir, True)
        cfg = DreamboothConfig(model_name=model_name, src=src, resolution=is_512)
        cfg.save()
        shutil.copytree(src, dest_dir)
