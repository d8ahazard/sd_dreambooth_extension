from __future__ import annotations

import collections
import os

from transformers import PretrainedConfig

try:
    from extensions.sd_dreambooth_extension.dreambooth import shared  # noqa
    from extensions.sd_dreambooth_extension.dreambooth.utils.utils import cleanup  # noqa
except:
    from dreambooth.dreambooth import shared  # noqa
    from dreambooth.dreambooth.utils.utils import cleanup  # noqa

checkpoints_list = {}
checkpoint_alisases = {}
checkpoints_loaded = collections.OrderedDict()

model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(shared.models_path, model_dir))


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""

    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


class CheckpointInfo:
    def __init__(self, filename):
        self.filename = filename
        abspath = os.path.abspath(filename)

        if shared.cmd_opts.ckpt_dir is not None and abspath.startswith(shared.cmd_opts.ckpt_dir):
            name = abspath.replace(shared.cmd_opts.ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(filename)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = model_hash(filename)

        self.sha256 = hashes.sha256_from_cache(self.filename, "checkpoint/" + name)
        self.shorthash = self.sha256[0:10] if self.sha256 else None

        self.title = name if self.shorthash is None else f'{name} [{self.shorthash}]'

        self.ids = [self.hash, self.model_name, self.title, name, f'{name} [{self.hash}]'] + (
            [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]'] if self.shorthash else [])

    def register(self):
        checkpoints_list[self.title] = self
        for id in self.ids:
            checkpoint_alisases[id] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, "checkpoint/" + self.name)
        if self.sha256 is None:
            return

        self.shorthash = self.sha256[0:10]

        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]']

        checkpoints_list.pop(self.title)
        self.title = f'{self.name} [{self.shorthash}]'
        self.register()

        return self.shorthash


def list_models():
    checkpoints_list.clear()
    checkpoint_alisases.clear()
    model_list = modelloader.load_models(model_path=model_path, command_path=shared.cmd_opts.ckpt_dir,
                                         ext_filter=[".ckpt", ".safetensors"], ext_blacklist=[".vae.safetensors"])

    cmd_ckpt = shared.cmd_opts.ckpt
    if os.path.exists(cmd_ckpt):
        checkpoint_info = CheckpointInfo(cmd_ckpt)
        checkpoint_info.register()

        shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
        print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {cmd_ckpt}",
              file=sys.stderr)

    for filename in model_list:
        checkpoint_info = CheckpointInfo(filename)
        checkpoint_info.register()


def get_model_snapshots(model_name: str):
    try:
        from extensions.sd_dreambooth_extension.dreambooth.dataclasses.db_config import from_file
    except:
        from dreambooth.dreambooth.dataclasses.db_config import from_file  # noqa

    result = None
    try:
        import gradio
        result = gradio.update(visible=True)
    except:
        pass
    if model_name == "" or model_name is None:
        return result
    config = from_file(model_name)
    snaps_path = os.path.join(config.model_dir, "snapshots")
    snaps = []
    if os.path.exists(snaps_path):
        for directory in os.listdir(snaps_path):
            if "checkpoint_" in directory:
                fullpath = os.path.join(snaps_path, directory)
                snaps.append(fullpath)
    return snaps


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_db_models():
    model_dir = shared.models_path
    out_dir = os.path.join(model_dir, "dreambooth")
    output = []
    if os.path.exists(out_dir):
        dirs = os.listdir(out_dir)
        for found in dirs:
            if os.path.isdir(os.path.join(out_dir, found)):
                output.append(found)
    return output


def get_lora_models():
    model_dir = shared.lora_models_path
    out_dir = model_dir
    output = [""]
    if os.path.exists(out_dir):
        dirs = os.listdir(out_dir)
        for found in dirs:
            if os.path.isfile(os.path.join(out_dir, found)):
                if "_txt.pt" not in found and ".pt" in found:
                    output.append(found)
    return output


def unload_system_models():
    try:
        import modules.shared
        if modules.shared.sd_model is not None:
            modules.shared.sd_model.to("cpu")
        for former in modules.shared.face_restorers:
            try:
                former.send_model_to("cpu")
            except:
                pass
        cleanup()
    except:
        pass


def reload_system_models():
    try:
        import modules.shared
        if modules.shared.sd_model is not None:
            modules.shared.sd_model.to(shared.device)
        print("Restored system models.")
    except:
        pass


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


# from modules.sd_models import CheckpointInfo

def get_checkpoint_match(search_string):
    try:
        from modules import sd_models
        for info in sd_models.checkpoints_list.values():
            if search_string in info.title or search_string in info.model_name or search_string in info.filename:
                return info
    except:
        pass
    return None


def disable_safe_unpickle():
    try:
        from modules import shared as auto_shared
        auto_shared.cmd_opts.disable_safe_unpickle = True
    except:
        pass


def enable_safe_unpickle():
    try:
        from modules import shared as auto_shared
        auto_shared.cmd_opts.disable_safe_unpickle = False
    except:
        pass
