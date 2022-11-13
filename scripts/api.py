import base64
import io
import time
import uvicorn
from fastapi import FastAPI
from dreambooth import conversion, dreambooth
import gradio as gr

import modules.script_callbacks as script_callbacks


def dreamBoothAPI(demo: gr.Blocks, app: FastAPI):
    @app.post("/dreambooth/createModel")
    async def createModel(name: str, source: str, scheduler: str):
        print("Creating new Checkpoint: " + name)
        fn = conversion.extract_checkpoint(name, source, scheduler)


script_callbacks.on_app_started(dreamBoothAPI)

print("Dreambooth API layer loaded")