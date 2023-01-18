import json
import os
import traceback
from typing import List, Dict

from pydantic import BaseModel

from extensions.sd_dreambooth_extension.dreambooth import db_shared as shared, db_shared
from extensions.sd_dreambooth_extension.dreambooth.db_concept import Concept

# Keys to save, replacing our dumb __init__ method
save_keys = []

# Keys to return to the ui when Load Settings is clicked.
ui_keys = []


def sanitize_name(name):
    return "".join(x for x in name if (x.isalnum() or x in "._- "))


class DreamboothConfig(BaseModel):
    adamw_weight_decay: float = 0.01
    attention: str = "default"
    cache_latents: bool = True
    center_crop: bool = True
    clip_skip: int = 1
    concepts_list: List[Dict] = []
    concepts_path: str = ""
    custom_model_name: str = ""
    epoch: int = 0
    epoch_pause_frequency: int = 0
    epoch_pause_time: int = 0
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    gradient_set_to_none: bool = True
    graph_smoothing: int = 50
    half_model: bool = False
    has_ema: bool = False
    hflip: bool = False
    initial_revision: int = 0
    learning_rate: float = 5e-6
    learning_rate_min: float = 1e-6
    lifetime_revision: int = 0
    lora_learning_rate: float = 1e-4
    lora_model_name: str = ""
    lora_rank: int = 4
    lora_txt_learning_rate: float = 5e-5
    lora_txt_weight: float = 1.0
    lora_weight: float = 1.0
    lr_cycles: int = 1
    lr_factor: float = 0.5
    lr_power: float = 1.0
    lr_scale_pos: float = 0.5
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 0
    max_token_length: int = 75
    mixed_precision: str = "fp16"
    model_name: str = ""
    model_dir: str = ""
    model_path: str = ""
    num_train_epochs: int = 100
    pad_tokens: bool = True
    pretrained_model_name_or_path: str = ""
    pretrained_vae_name_or_path: str = ""
    prior_loss_scale: bool = False
    prior_loss_target: int = 100
    prior_loss_weight: float = 1.0
    prior_loss_weight_min: float = 0.1
    resolution: int = 512
    revision: int = 0
    sample_batch_size: int = 1
    sanity_prompt: str = ""
    sanity_seed: int = 420420
    save_ckpt_after: bool = True
    save_ckpt_cancel: bool = False
    save_ckpt_during: bool = True
    save_embedding_every: int = 25
    save_lora_after: bool = True
    save_lora_cancel: bool = False
    save_lora_during: bool = True
    save_preview_every: int = 5
    save_safetensors: bool = False
    save_state_after: bool = False
    save_state_cancel: bool = False
    save_state_during: bool = False
    scheduler: str = "ddim"
    shuffle_tags: bool = False
    snapshot: str = ""
    src: str = ""
    stop_text_encoder: float = 1.0
    train_batch_size: int = 1
    train_imagic: bool = False
    train_unet: bool = True
    use_8bit_adam: bool = True
    use_concepts: bool = False
    use_ema: bool = True
    use_lora: bool = False
    use_subdir: bool = False
    v2: bool = False

    def __init__(self, model_name: str = "", scheduler: str = "ddim", v2: bool = False, src: str = "",
                 resolution: int = 512, **kwargs):

        super().__init__(**kwargs)
        model_name = sanitize_name(model_name)
        models_path = shared.dreambooth_models_path
        if models_path == "" or models_path is None:
            models_path = os.path.join(shared.models_path, "dreambooth")
        model_dir = os.path.join(models_path, model_name)
        working_dir = os.path.join(model_dir, "working")

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        self.model_name = model_name
        self.model_dir = model_dir
        self.pretrained_model_name_or_path = working_dir
        self.resolution = resolution
        self.src = src
        self.scheduler = scheduler
        if v2 == 'True':
            self.v2 = True
        elif v2 == 'False':
            self.v2 = False
        else:
            self.v2 = v2

    # Actually save as a file
    def save(self, backup=False):
        """
        Save the config file
        """
        models_path = shared.dreambooth_models_path
        if models_path == "" or models_path is None:
            models_path = os.path.join(shared.models_path, "dreambooth")

        config_file = os.path.join(models_path, self.model_name, "db_config.json")
        if backup:
            backup_dir = os.path.join(models_path, self.model_name, "backups")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            config_file = os.path.join(models_path, self.model_name, "backups", f"db_config_{self.revision}.json")
        with open(config_file, "w") as outfile:
            json.dump(self.__dict__, outfile, indent=4)

    def load_params(self, params_dict):
        for key, value in params_dict.items():
            if "db_" in key:
                key = key.replace("db_", "")
            if hasattr(self, key):
                setattr(self, key, value)

    # Pass a dict and return a list of Concept objects
    def concepts(self, required: int = -1):
        concepts = []
        c_idx = 0
        if required == -1:
            required = len(self.concepts_list)
        print(f"Four concepts required, we have {len(self.concepts_list)} currently.")
        for concept_dict in self.concepts_list:
            concept = Concept(input_dict=concept_dict)
            if concept.is_valid:
                if concept.class_data_dir == "" or concept.class_data_dir is None:
                    concept.class_data_dir = os.path.join(self.model_dir, f"classifiers_{c_idx}")
                concepts.append(concept)
                c_idx += 1
            else:
                print(f"Invalid concept: {concept.instance_data_dir}")

        missing = len(concepts) - required
        if missing > 0:
            print(f"Adding {missing} blank concepts?")
            concepts.extend([Concept(None)] * missing)
        return concepts

    # Set default values
    def check_defaults(self):
        if self.model_name is not None and self.model_name != "":
            if self.revision == "" or self.revision is None:
                self.revision = 0
            if self.epoch == "" or self.epoch is None:
                self.epoch = 0
            self.model_name = "".join(x for x in self.model_name if (x.isalnum() or x in "._- "))
            models_path = shared.dreambooth_models_path
            if models_path == "" or models_path is None:
                models_path = os.path.join(shared.models_path, "dreambooth")
            model_dir = os.path.join(models_path, self.model_name)
            working_dir = os.path.join(model_dir, "working")
            if not os.path.exists(working_dir):
                os.makedirs(working_dir)
            self.model_dir = model_dir
            self.pretrained_model_name_or_path = working_dir


def save_config(*args):
    params = list(args)
    concept_keys = ["c1_", "c2_", "c3_", "c4_"]
    model_name = params[0]
    if model_name is None or model_name == "":
        print("Invalid model name.")
        return
    config = from_file(model_name)
    if config is None:
        config = DreamboothConfig(model_name)
    params_dict = dict(zip(save_keys, params))
    concepts_list = []
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

    config.load_params(params_dict)

    print("Saved settings.")
    config.save()


def from_file(model_name):
    """
    Load config data from UI
    Args:
        model_name: The config to load

    Returns: Dict | None

    """
    if model_name == "" or model_name is None:
        return None

    model_name = sanitize_name(model_name)
    models_path = db_shared.dreambooth_models_path
    if models_path == "" or models_path is None:
        models_path = os.path.join(db_shared.models_path, "dreambooth")
    config_file = os.path.join(models_path, model_name, "db_config.json")
    try:
        with open(config_file, 'r') as openfile:
            config_dict = json.load(openfile)

        config = DreamboothConfig(model_name)
        config.load_params(config_dict)
        return config
    except Exception as e:
        print(f"Exception loading config: {e}")
        traceback.print_exc()
        return None
