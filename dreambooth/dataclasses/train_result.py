from PIL import Image

from dreambooth.dataclasses.db_config import DreamboothConfig


class TrainResult:
    config: DreamboothConfig = None
    msg: str = ""
    samples: [Image] = []
