from PIL import Image

try:
    from extensions.sd_dreambooth_extension.dreambooth.dataclasses.db_config import DreamboothConfig
except:
    from dreambooth.dreambooth.dataclasses.db_config import DreamboothConfig  # noqa


class TrainResult:
    config: DreamboothConfig = None
    msg: str = ""
    samples: [Image] = []
