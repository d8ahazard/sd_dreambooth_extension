import hashlib
import json
import os

from pydantic import Field

from dreambooth import shared
from finetune.dataclasses.base_config import BaseConfig
from modules.sd_models import CheckpointInfo


class WorkspaceConfig(BaseConfig):
    name: str = Field("workspace", title="Workspace name", description="Name of the workspace")
    base_model: str = Field("", title="Base model", description="Base model to use for training")
    base_model_type: str = Field("v1", title="Base model type", description="Base model type to use for training")
    base_model_src: str = Field("local", title="Base model source", description="Either 'local' for a user-provided model, or a URL to a HF Model")
    base_model_hash: str = Field("", title="Base model hash", description="Hash of the base model")

    def __init__(self, **data):
        super().__init__(**data)
        self.compute_hash()

    def compute_hash(self):
        print(f"Computing hash for {self.base_model_src}")
        if self.base_model_src == 'local':
            if self.base_model_hash == "":
                # Calculate the sha265 hash of the model
                if not os.path.exists(self.base_model):
                    return
                ckpt_info = CheckpointInfo(self.base_model)
                self.base_model_hash = ckpt_info.calculate_shorthash()
        elif self.base_model_src != '' and self.base_model_hash == "":
            self.base_model_hash = hashlib.sha256(self.base_model_src.encode('utf-8')).hexdigest()
        print(f"Hash is {self.base_model_hash}")


