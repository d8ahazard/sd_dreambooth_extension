import os

from fastapi import FastAPI, Query

import scripts.api
from core.handlers.websocket import SocketHandler
from core.modules.base.module_base import BaseModule


class DreamboothModule(BaseModule):
    def __init__(self):
        self.name: str = "Dreambooth"
        self.path = os.path.abspath(os.path.dirname(__file__))
        super().__init__(self.name, self.path)
        socket_handler = SocketHandler()
        socket_handler.register("train_dreambooth", self._train_dreambooth)

    async def _train_dreambooth(self, data):
        self.logger.debug(f"Train dreambooth called: {data}")

    def initialize(self, app: FastAPI, handler: SocketHandler):
        self._initialize_api(app)

    def _initialize_api(self, app: FastAPI):
        return scripts.api.dreambooth_api(None, app)
