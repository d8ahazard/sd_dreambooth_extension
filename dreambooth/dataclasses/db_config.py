import json
import logging
import os
import traceback
from pathlib import Path
from typing import List, Dict

from pydantic import BaseModel, Field

from dreambooth import shared  # noqa
from dreambooth.dataclasses.base_config import BaseConfig
from dreambooth.dataclasses.db_concept import Concept  # noqa
from dreambooth.utils.image_utils import get_scheduler_names  # noqa
from dreambooth.utils.utils import list_attention

# Keys to save, replacing our dumb __init__ method
save_keys = []

# Keys to return to the ui when Load Settings is clicked.
ui_keys = []


def sanitize_name(name):
    return "".join(x for x in name if (x.isalnum() or x in "._- "))


class DreamboothConfig(BaseConfig):
    config_prefix: str = Field("db", description="Prefix for the config file.")
    attention: str = Field("xformers", description="Attention model.")
    cache_latents: bool = Field(True, description="Cache latents.")
    clip_skip: int = Field(1, description="Clip skip.")
    concepts_list: List[Dict] = Field([], description="Concepts list.")
    concepts_path: str = Field("", description="Path to the concepts.")
    custom_model_name: str = Field("", description="Custom model name.")
    deterministic: bool = Field(False, description="Deterministic mode.")
    disable_class_matching: bool = Field(False, description="Disable class matching.")
    disable_logging: bool = Field(False, description="Disable logging.")
    ema_predict: bool = Field(False, description="EMA predict.")
    epoch: int = Field(0, description="Current epoch.")
    epoch_pause_frequency: int = Field(0, description="Epoch pause frequency.")
    epoch_pause_time: int = Field(0, description="Epoch pause time.")
    freeze_clip_normalization: bool = Field(False, description="Freeze clip normalization.")
    gradient_accumulation_steps: int = Field(1, description="Gradient accumulation steps.")
    gradient_checkpointing: bool = Field(True, description="Gradient checkpointing.")
    gradient_set_to_none: bool = Field(True, description="Gradient set to none.")
    graph_smoothing: int = Field(50, description="Graph smoothing.")
    half_model: bool = Field(False, description="Half model.")
    has_ema: bool = Field(False, description="Has EMA.")
    hflip: bool = Field(False, description="Horizontal flip.")
    infer_ema: bool = Field(False, description="Infer EMA.")
    initial_revision: int = Field(0, description="Initial revision.")
    learning_rate: float = Field(5e-6, description="Learning rate.")
    learning_rate_min: float = Field(1e-6, description="Minimum learning rate.")
    lifetime_revision: int = Field(0, description="Lifetime revision.")
    lora_learning_rate: float = Field(1e-4, description="LoRA learning rate.")
    lora_model_name: str = Field("", description="LoRA model name.")
    lora_txt_learning_rate: float = Field(5e-5, description="LoRA text learning rate.")
    lora_txt_rank: int = Field(4, description="LoRA text rank.")
    lora_txt_weight: float = Field(1.0, description="LoRA text weight.")
    lora_unet_rank: int = Field(4, description="LoRA UNet rank.")
    lora_weight: float = Field(1.0, description="LoRA weight.")
    lora_use_buggy_requires_grad: bool = Field(False, description="LoRA use buggy requires grad.")
    lr_num_cycles: int = Field(1, description="Learning rate cycles.")
    lr_factor: float = Field(0.5, description="Learning rate factor.")
    lr_power: float = Field(1.0, description="Learning rate power.")
    lr_scale_pos: float = Field(0.5, description="Learning rate scale position.")
    lr_scheduler: str = Field("constant_with_warmup", description="Learning rate scheduler.")
    lr_warmup_steps: int = Field(0, description="Learning rate warmup steps.")
    max_token_length: int = Field(75, description="Max token length.")
    mixed_precision: str = Field("fp16", description="Mixed precision mode.")
    model_dir: str = Field("", description="Model directory.")
    model_name: str = Field("", description="Model name.")
    model_path: str = Field("", description="Model path.")
    noise_scheduler: str = Field("DDPM", description="Noise scheduler.")
    num_train_epochs: int = Field(100, description="Number of training epochs.")
    offset_noise: float = Field(0, description="Offset noise.")
    optimizer: str = Field("8bit AdamW", description="Optimizer.")
    pad_tokens: bool = Field(True, description="Pad tokens.")
    pretrained_model_name_or_path: str = Field("", description="Pretrained model name or path.")
    pretrained_vae_name_or_path: str = Field("", description="Pretrained VAE model name or path.")
    prior_loss_scale: bool = Field(False, description="Prior loss scale.")
    prior_loss_target: int = Field(100, description="Prior loss target.")
    prior_loss_weight: float = Field(0.75, description="Prior loss weight.")
    prior_loss_weight_min: float = Field(0.1, description="Minimum prior loss weight.")
    resolution: int = Field(512, description="Resolution.")
    revision: int = Field(0, description="Revision.")
    sample_batch_size: int = Field(1, description="Sample batch size.")
    sanity_prompt: str = Field("", description="Sanity prompt.")
    sanity_seed: int = Field(420420, description="Sanity seed.")
    save_ckpt_after: bool = Field(True, description="Save checkpoint after.")
    save_ckpt_cancel: bool = Field(False, description="Cancel saving of checkpoint.")
    save_ckpt_during: bool = Field(True, description="Save checkpoint during.")
    save_ema: bool = Field(True, description="Save EMA.")
    save_embedding_every: int = Field(25, description="How often to save weights.")
    save_lora_after: bool = Field(True, description="Save LoRA after.")
    save_lora_cancel: bool = Field(False, description="Cancel saving of LoRA.")
    save_lora_during: bool = Field(True, description="Save LoRA during.")
    save_lora_for_extra_net: bool = Field(True, description="Save LoRA for extra net.")
    save_preview_every: int = Field(5, description="Save preview every.")
    save_state_after: bool = Field(False, description="Save state after.")
    save_state_cancel: bool = Field(False, description="Cancel saving of state.")
    save_state_during: bool = Field(False, description="Save state during.")
    scheduler: str = Field("ddim", description="Scheduler.")
    shared_diffusers_path: str = Field("", description="Shared diffusers path.")
    shuffle_tags: bool = Field(True, description="Shuffle tags.")
    snapshot: str = Field("", description="Snapshot.")
    split_loss: bool = Field(True, description="Split loss.")
    src: str = Field("", description="The source checkpoint.")
    stop_text_encoder: float = Field(1.0, description="Stop text encoder.")
    strict_tokens: bool = Field(False, description="Strict tokens.")
    dynamic_img_norm: bool = Field(False, description="Dynamic image normalization.")
    tenc_weight_decay: float = Field(0.01, description="Text encoder weight decay.")
    tenc_grad_clip_norm: float = Field(0.00, description="Text encoder gradient clipping norm.")
    tomesd: float = Field(0, description="TomesD.")
    train_batch_size: int = Field(1, description="Training batch size.")
    train_imagic: bool = Field(False, description="Train iMagic.")
    train_unet: bool = Field(True, description="Train UNet.")
    train_unfrozen: bool = Field(True, description="Train unfrozen.")
    txt_learning_rate: float = Field(5e-6, description="Text learning rate.")
    use_concepts: bool = Field(False, description="Use concepts.")
    use_ema: bool = Field(True, description="Use EMA.")
    use_lora: bool = Field(False, description="Use LoRA.")
    use_lora_extended: bool = Field(False, description="Use LoRA extended.")
    use_shared_src: bool = Field(False, description="Use shared source.")
    use_subdir: bool = Field(False, description="Use subdirectory.")
    v2: bool = Field(False, description="If this is a V2 Model or not.")
    weight_decay: float = Field(0.01, description="Weight decay.")

    def __init__(
            self,
            **kwargs
    ):

        super().__init__(**kwargs)
        if "model_name" in kwargs:
            model_name = kwargs["model_name"]
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
        if "resolution" in kwargs:
            self.resolution = kwargs["resolution"]
        if "v2" in kwargs:
            self.v2 = kwargs["v2"]
        if "src" in kwargs:
            self.src = kwargs["src"]
        if "scheduler" in kwargs:
            self.scheduler = kwargs["scheduler"]

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

    def load_from_file(self, model_dir=None):
        """
        Load config data from UI
        Args:
            model_dir: If specified, override the default model directory

        Returns: DreamboothConfig | None

        """
        config_file = os.path.join(model_dir, "db_config.json")
        try:
            with open(config_file, 'r') as openfile:
                config_dict = json.load(openfile)
            super().load_from_file(model_dir)
            self.load_params(config_dict)
            return self
        except Exception as e:
            print(f"Exception loading config: {e}")
            traceback.print_exc()
            return None


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
        config = DreamboothConfig(model_name=model_name)
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

        config = DreamboothConfig(model_name=model_name)
        config.load_params(config_dict)
        shared.db_model_config = config
        return config
    except Exception as e:
        print(f"Exception loading config: {e}")
        traceback.print_exc()
        return None
