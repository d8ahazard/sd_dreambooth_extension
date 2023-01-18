import json
import os
import traceback

from extensions.sd_dreambooth_extension.dreambooth import db_shared as shared, db_shared
from extensions.sd_dreambooth_extension.dreambooth.db_concept import Concept

# Keys to save, replacing our dumb __init__ method
save_keys = []

# Keys to return to the ui when Load Settings is clicked.
ui_keys = []

param_defaults = {"model_name": "", "attention": "default", "cache_latents": True, "center_crop": True, "clip_skip": 1,
    "concepts_path": "", "concepts_list": [], "custom_model_name": "", "epoch": 0, "epoch_pause_frequency": 0,
    "epoch_pause_time": 0, "gradient_accumulation_steps": 1, "gradient_checkpointing": True,
    "gradient_set_to_none": True, "graph_smoothing": 50, "half_model": False, "has_ema": False, "hflip": False,
    "initial_revision": 0, "learning_rate": 5e-6, "learning_rate_min": 1e-6, "lifetime_revision": 0,
    "lora_learning_rate": 1e-4, "lora_model_name": "", "lora_rank": 4, "lora_txt_learning_rate": 5e-5,
    "lora_txt_weight": 1.0, "lora_weight": 1.0, "lr_cycles": 1, "lr_factor": 0.5, "lr_power": 1.0, "lr_scale_pos": 0.5,
    "lr_scheduler": "constant_with_warmup", "lr_warmup_steps": 0, "max_token_length": 75, "mixed_precision": "fp16",
    "adamw_weight_decay": 0.01, "model_path": "", "num_train_epochs": 100, "pad_tokens": True,
    "pretrained_vae_name_or_path": "", "prior_loss_scale": False, "prior_loss_target": 100, "prior_loss_weight": 1.0,
    "prior_loss_weight_min": 0.1, "resolution": 512, "revision": 0, "sample_batch_size": 1, "sanity_prompt": "",
    "sanity_seed": 420420, "save_ckpt_after": True, "save_ckpt_cancel": False, "save_ckpt_during": True,
    "save_embedding_every": 25, "save_lora_after": True, "save_lora_cancel": False, "save_lora_during": True,
    "save_preview_every": 5, "save_safetensors": False, "save_state_after": False, "save_state_cancel": False,
    "save_state_during": False, "scheduler": "ddim", "src": "", "shuffle_tags": False, "snapshot": "",
    "train_batch_size": 1, "train_imagic": False, "stop_text_encoder": 1.0, "use_8bit_adam": True,
    "use_concepts": False, "use_ema": True, "use_lora": False, "use_subdir": False, "v2": False}


def sanitize_name(name):
    return "".join(x for x in name if (x.isalnum() or x in "._- "))


class DreamboothConfig:
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

    # Take a list of params from the UI, load, save
    # Pass a dict of values to set attributes
    def load_params(self, params_dict=None):
        if params_dict is None:
            params_dict = param_defaults
        for key, value in params_dict.items():
            if "db_" in key:
                key = key.replace("db_", "")
            if key == "concepts_list":
                try:
                    value = self.load_concepts(value)
                except Exception as e:
                    print(f"Exception loading concepts: {e}")
            if hasattr(self, key):
                setattr(self, key, value)

    # Pass a dict and return a list of Concept objects
    def load_concepts(self, concepts_list):
        concepts = []
        c_idx = 0
        for concept_dict in concepts_list:
            concept = Concept(input_dict=concept_dict)
            if concept.is_valid():
                if "class_data_dir" not in concept.__dict__ or concept.class_data_dir == "" or concept.class_data_dir is None:
                    concept.class_data_dir = os.path.join(self.model_dir, f"classifiers_{c_idx}")
                concepts.append(concept)
                c_idx += 1
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

    def __init__(self, model_name: str = "", scheduler: str = "ddim", v2: bool = False, src: str = "",
                 resolution: int = 512):

        model_name = sanitize_name(model_name)
        models_path = shared.dreambooth_models_path
        if models_path == "" or models_path is None:
            models_path = os.path.join(shared.models_path, "dreambooth")
        model_dir = os.path.join(models_path, model_name)
        working_dir = os.path.join(model_dir, "working")

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        self.attention = None
        self.cache_latents = None
        self.center_crop = None
        self.clip_skip = None
        self.concepts_path = None
        self.concepts_list = None
        self.custom_model_name = None
        self.epoch = None
        self.epoch_pause_frequency = None
        self.epoch_pause_time = None
        self.gradient_accumulation_steps = None
        self.gradient_checkpointing = None
        self.gradient_set_to_none = None
        self.graph_smoothing = None
        self.half_model = None
        self.has_ema = None
        self.hflip = None
        self.initial_revision = None
        self.learning_rate = None
        self.learning_rate_min = None
        self.lifetime_revision = None
        self.lora_learning_rate = None
        self.lora_model_name = None
        self.lora_rank = None
        self.lora_txt_learning_rate = None
        self.lora_txt_weight = None
        self.lora_weight = None
        self.lr_cycles = None
        self.lr_factor = None
        self.lr_power = None
        self.lr_scale_pos = None
        self.lr_scheduler = None
        self.lr_warmup_steps = None
        self.max_token_length = None
        self.mixed_precision = None
        self.adamw_weight_decay = None
        self.num_train_epochs = None
        self.pad_tokens = None
        self.pretrained_vae_name_or_path = None
        self.prior_loss_scale = None
        self.prior_loss_target = None
        self.prior_loss_weight = None
        self.prior_loss_weight_min = None
        self.revision = None
        self.sample_batch_size = None
        self.sanity_prompt = None
        self.sanity_seed = None
        self.save_ckpt_after = None
        self.save_ckpt_cancel = None
        self.save_ckpt_during = None
        self.save_embedding_every = None
        self.save_lora_after = None
        self.save_lora_cancel = None
        self.save_lora_during = None
        self.save_preview_every = None
        self.save_safetensors = None
        self.save_state_after = None
        self.save_state_cancel = None
        self.save_state_during = None
        self.shuffle_tags = None
        self.snapshot = None
        self.train_batch_size = None
        self.train_imagic = None
        self.train_unet = None 
        self.stop_text_encoder = None
        self.use_8bit_adam = None
        self.use_concepts = None
        self.use_ema = None
        self.use_lora = None
        self.use_subdir = None

        self.load_params()

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


def save_config(*args):
    params = list(args)
    model_name = params[0]
    if model_name is None or model_name == "":
        print("Invalid model name.")
        return
    config = from_file(model_name)
    if config is None:
        config = DreamboothConfig(model_name)
    params_dict = dict(zip(save_keys, params))
    print("Applying new params...")
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
        print("Config created, loading params...")
        config.load_params(config_dict)
        return config
    except Exception as e:
        print(f"Exception loading config: {e}")
        traceback.print_exc()
        return None
