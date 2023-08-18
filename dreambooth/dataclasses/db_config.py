import json
import logging
import os
import traceback
from pathlib import Path
from typing import List, Dict

from pydantic import BaseModel

from dreambooth import shared  # noqa
from dreambooth.dataclasses.db_concept import Concept  # noqa
from dreambooth.utils.image_utils import get_scheduler_names  # noqa
from dreambooth.utils.utils import list_attention

# Keys to save, replacing our dumb __init__ method
save_keys = []

# Keys to return to the ui when Load Settings is clicked.
ui_keys = []


def sanitize_name(name):
    return "".join(x for x in name if (x.isalnum() or x in "._- "))


class DreamboothConfig(BaseModel):
    # These properties MUST be sorted alphabetically
    weight_decay: float = 0.01
    attention: str = "xformers"
    cache_latents: bool = True
    clip_skip: int = 1
    concepts_list: List[Dict] = []
    concepts_path: str = ""
    custom_model_name: str = ""
    deterministic: bool = False
    disable_class_matching: bool = False
    disable_logging: bool = False
    ema_predict: bool = False
    epoch: int = 0
    epoch_pause_frequency: int = 0
    epoch_pause_time: int = 0
    freeze_clip_normalization: bool = False
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    gradient_set_to_none: bool = True
    graph_smoothing: int = 50
    half_model: bool = False
    has_ema: bool = False
    hflip: bool = False
    infer_ema: bool = False
    initial_revision: int = 0
    learning_rate: float = 5e-6
    learning_rate_min: float = 1e-6
    lifetime_revision: int = 0
    lora_learning_rate: float = 1e-4
    lora_model_name: str = ""
    lora_txt_learning_rate: float = 5e-5
    lora_txt_rank: int = 4
    lora_txt_weight: float = 1.0
    lora_unet_rank: int = 4
    lora_weight: float = 1.0
    lora_use_buggy_requires_grad: bool = False
    lr_cycles: int = 1
    lr_factor: float = 0.5
    lr_power: float = 1.0
    lr_scale_pos: float = 0.5
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 0
    max_token_length: int = 75
    mixed_precision: str = "fp16"
    model_dir: str = ""
    model_name: str = ""
    model_path: str = ""
    model_type: str = "v1x"
    noise_scheduler: str = "DDPM"
    num_train_epochs: int = 100
    offset_noise: float = 0
    optimizer: str = "8bit AdamW"
    pad_tokens: bool = True
    pretrained_model_name_or_path: str = ""
    pretrained_vae_name_or_path: str = ""
    prior_loss_scale: bool = False
    prior_loss_target: int = 100
    prior_loss_weight: float = 0.75
    prior_loss_weight_min: float = 0.1
    resolution: int = 512
    revision: int = 0
    sample_batch_size: int = 1
    sanity_prompt: str = ""
    sanity_seed: int = 420420
    save_ckpt_after: bool = True
    save_ckpt_cancel: bool = False
    save_ckpt_during: bool = True
    save_ema: bool = True
    save_embedding_every: int = 25
    save_lora_after: bool = True
    save_lora_cancel: bool = False
    save_lora_during: bool = True
    save_lora_for_extra_net: bool = True
    save_preview_every: int = 5
    save_safetensors: bool = True
    save_state_after: bool = False
    save_state_cancel: bool = False
    save_state_during: bool = False
    scheduler: str = "ddim"
    shared_diffusers_path: str = ""
    shuffle_tags: bool = True
    snapshot: str = ""
    split_loss: bool = True
    src: str = ""
    stop_text_encoder: float = 1.0
    strict_tokens: bool = False
    dynamic_img_norm: bool = False
    tenc_weight_decay: float = 0.01
    tenc_grad_clip_norm: float = 0.00
    tomesd: float = 0
    train_batch_size: int = 1
    train_imagic: bool = False
    train_unet: bool = True
    train_unfrozen: bool = True
    txt_learning_rate: float = 5e-6
    use_concepts: bool = False
    use_ema: bool = True
    use_lora: bool = False
    use_lora_extended: bool = False
    use_shared_src: bool = False,
    use_subdir: bool = False
    v2: bool = False

    def __init__(
            self,
            model_name: str = "",
            model_dir: str = "",
            v2: bool = False,
            src: str = "",
            resolution: int = 512,
            **kwargs
    ):

        super().__init__(**kwargs)

        model_name = sanitize_name(model_name)
        if "models_path" in kwargs:
            models_path = kwargs["models_path"]
            print(f"Using models path: {models_path}")
        else:
            models_path = shared.dreambooth_models_path
            if models_path == "" or models_path is None:
                models_path = os.path.join(shared.models_path, "dreambooth")

            # If we're using the new UI, this should be populated, so load models from here.
            if len(shared.paths):
                models_path = os.path.join(shared.paths["models"], "dreambooth")

        if not self.use_lora:
            self.lora_model_name = ""

        model_dir = os.path.join(models_path, model_name)
        # print(f"Model dir set to: {model_dir}")
        working_dir = os.path.join(model_dir, "working")

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        self.model_name = model_name
        self.model_dir = model_dir
        self.pretrained_model_name_or_path = working_dir
        self.resolution = resolution
        self.src = src
        self.scheduler = "ddim"
        self.v2 = v2

    # Actually save as a file
    def save(self, backup=False):
        """
        Save the config file
        """
        models_path = self.model_dir
        logger = logging.getLogger(__name__)
        logger.debug("Saving to %s", models_path)

        if os.name == 'nt' and '/' in models_path:
            # replace linux path separators with windows path separators
            models_path = models_path.replace('/', '\\')
        elif os.name == 'posix' and '\\' in models_path:
            # replace windows path separators with linux path separators
            models_path = models_path.replace('\\', '/')
        self.model_dir = models_path
        config_file = os.path.join(models_path, "db_config.json")

        if backup:
            backup_dir = os.path.join(models_path, "backups")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            config_file = os.path.join(models_path, "backups", f"db_config_{self.revision}.json")

        with open(config_file, "w") as outfile:
            json.dump(self.__dict__, outfile, indent=4)

    def load_params(self, params_dict):
        sched_swap = False
        for key, value in params_dict.items():
            if "db_" in key:
                key = key.replace("db_", "")
            if key == "attention" and value == "flash_attention":
                value = list_attention()[-1]
                print(f"Replacing flash attention in config to {value}")

            if key == "scheduler":
                schedulers = get_scheduler_names()
                if value not in schedulers:
                    sched_swap = True
                    for scheduler in schedulers:
                        if value.lower() in scheduler.lower():
                            print(f"Updating scheduler name to: {scheduler}")
                            value = scheduler
                            break

            if hasattr(self, key):
                key, value = self.validate_param(key, value)
                setattr(self, key, value)
        if sched_swap:
            self.save()

    @staticmethod
    def validate_param(key, value):
        replaced_params = {
            # "old_key" : {
            #   "new_key": "...",
            #   "values": [{
            #       "old": ["...", "..."]
            #       "new": "..."
            #   }]
            # }
            "weight_decay": {
                "new_key": "weight_decay",
            },
            "deis_train_scheduler": {
                "new_key": "noise_scheduler",
                "values": [{
                    "old": [True],
                    "new": "DDPM"
                }],
            },
            "optimizer": {
                "values": [{
                    "old": ["8Bit Adam"],
                    "new": "8bit AdamW"
                }],
            },
            "save_safetensors": {
                "values": [{
                    "old": [False],
                    "new": True
                }],
            }
        }

        if key in replaced_params.keys():
            replacement = replaced_params[key]
            if "new_key" in replacement:
                key = replacement["new_key"]
            if "values" in replacement:
                for _value in replacement["values"]:
                    if value in _value["old"]:
                        value = _value["new"]
        return key, value

    # Pass a dict and return a list of Concept objects
    def concepts(self, required: int = -1):
        concepts = []
        c_idx = 0
        # If using a file for concepts and not requesting from UI, load from file
        if self.use_concepts and self.concepts_path and required == -1:
            concepts_list = concepts_from_file(self.concepts_path)

        # Otherwise, use 'stored' list
        else:
            concepts_list = self.concepts_list
        if required == -1:
            required = len(concepts_list)

        for concept_dict in concepts_list:
            concept = Concept(input_dict=concept_dict)
            if concept.is_valid:
                if concept.class_data_dir == "" or concept.class_data_dir is None:
                    concept.class_data_dir = os.path.join(self.model_dir, f"classifiers_{c_idx}")
                concepts.append(concept)
                c_idx += 1

        missing = len(concepts) - required
        if missing > 0:
            concepts.extend([Concept(None)] * missing)
        return concepts

    def refresh(self):
        """
        Reload self from file

        """
        models_path = shared.dreambooth_models_path
        if models_path == "" or models_path is None:
            models_path = os.path.join(shared.models_path, "dreambooth")
        config_file = os.path.join(models_path, self.model_name, "db_config.json")
        try:
            with open(config_file, 'r') as openfile:
                config_dict = json.load(openfile)

            self.load_params(config_dict)
            shared.db_model_config = self
        except Exception as e:
            print(f"Exception loading config: {e}")
            traceback.print_exc()
            return None

    def get_pretrained_model_name_or_path(self):
        if self.shared_diffusers_path != "" and not self.use_lora:
            raise Exception(f"shared_diffusers_path is \"{self.shared_diffusers_path}\" but use_lora is false")
        if self.shared_diffusers_path != "":
            return self.shared_diffusers_path
        if not self.pretrained_model_name_or_path or self.pretrained_model_name_or_path == "":
            return os.path.join(self.model_dir, "working")
        return self.pretrained_model_name_or_path


def concepts_from_file(concepts_path: str):
    concepts = []
    if os.path.exists(concepts_path) and os.path.isfile(concepts_path):
        try:
            with open(concepts_path, "r") as concepts_file:
                concepts_str = concepts_file.read()
        except Exception as e:
            print(f"Exception opening concepts file: {e}")
    else:
        concepts_str = concepts_path

    try:
        concepts_data = json.loads(concepts_str)
        for concept_data in concepts_data:
            concepts_path_dir = Path(concepts_path).parent # Get which folder is JSON file reside
            instance_data_dir = concept_data.get("instance_data_dir")
            if not os.path.isabs(instance_data_dir):
                print(f"Rebuilding portable concepts path: {concepts_path_dir} + {instance_data_dir}")
                concept_data["instance_data_dir"] = os.path.join(concepts_path_dir, instance_data_dir)

            concept = Concept(input_dict=concept_data)
            if concept.is_valid:
                concepts.append(concept.__dict__)
    except Exception as e:
        print(f"Exception parsing concepts: {e}")
    print(f"Loaded concepts: {concepts}")
    return concepts


def save_config(*args):
    params = list(args)
    concept_keys = ["c1_", "c2_", "c3_", "c4_"]
    params_dict = dict(zip(save_keys, params))
    concepts_list = []
    # If using a concepts file/string, keep concepts_list empty.
    if params_dict["db_use_concepts"] and params_dict["db_concepts_path"]:
        concepts_list = []
        params_dict["concepts_list"] = concepts_list
    else:
        for concept_key in concept_keys:
            concept_dict = {}
            for key, param in params_dict.items():
                if concept_key in key and param is not None:
                    concept_dict[key.replace(concept_key, "")] = param
            concept_test = Concept(concept_dict)
            if concept_test.is_valid:
                concepts_list.append(concept_test.__dict__)
        existing_concepts = params_dict["concepts_list"] if "concepts_list" in params_dict else []
        if len(concepts_list) and not len(existing_concepts):
            params_dict["concepts_list"] = concepts_list

    model_name = params_dict["db_model_name"]
    if model_name is None or model_name == "":
        print("Invalid model name.")
        return

    config = from_file(model_name)
    if config is None:
        config = DreamboothConfig(model_name)
    config.load_params(params_dict)
    shared.db_model_config = config
    config.save()


def from_file(model_name, model_dir=None):
    """
    Load config data from UI
    Args:
        model_name: The config to load
        model_dir: If specified, override the default model directory

    Returns: Dict | None

    """
    if isinstance(model_name, list) and len(model_name) > 0:
        model_name = model_name[0]

    if model_name == "" or model_name is None:
        return None

    #model_name = sanitize_name(model_name)
    if model_dir:
        models_path = model_dir
        shared.dreambooth_models_path = models_path
    else:
        models_path = shared.dreambooth_models_path
        if models_path == "" or models_path is None:
            models_path = os.path.join(shared.models_path, "dreambooth")
    config_file = os.path.join(models_path, model_name, "db_config.json")
    try:
        with open(config_file, 'r') as openfile:
            config_dict = json.load(openfile)

        config = DreamboothConfig(model_name)
        config.load_params(config_dict)
        shared.db_model_config = config
        return config
    except Exception as e:
        print(f"Exception loading config: {e}")
        traceback.print_exc()
        return None
