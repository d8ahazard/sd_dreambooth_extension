from PIL import Image

from extensions.sd_dreambooth_extension.dreambooth.dataclasses.db_config import DreamboothConfig


class TrainResult:
    config: DreamboothConfig = None
    msg: str = ""
    samples: [Image] = []
