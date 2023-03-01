from asyncio import Lock

import uvicorn
from fastapi import FastAPI, APIRouter


class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock):
        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    def launch(self, server_name, port):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name, port=port)
