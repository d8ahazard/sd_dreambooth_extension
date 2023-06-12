import json
import logging
import os
import traceback
from pathlib import Path
from typing import List, Dict, Optional

from pydantic import BaseModel, Field

from dreambooth import shared  # noqa
from dreambooth.dataclasses.base_config import BaseConfig
from dreambooth.dataclasses.db_concept import Concept  # noqa
from dreambooth.utils.image_utils import get_scheduler_names  # noqa
from dreambooth.utils.utils import list_attention, list_precisions

# Keys to save, replacing our dumb __init__ method
save_keys = []

# Keys to return to the ui when Load Settings is clicked.
ui_keys = []


def sanitize_name(name):
    return "".join(x for x in name if (x.isalnum() or x in "._- "))


class TrainingConfig(BaseConfig):
    precisions = list_precisions()
    attentions = list_attention()
    adam_beta1: float = Field(0.9, description="The beta1 parameter for the Adam optimizer.")
    adam_beta2: float = Field(0.999, description="The beta2 parameter for the Adam optimizer.")
    adam_epsilon: float = Field(1e-08, description="Epsilon value for the Adam optimizer.")
    adam_weight_decay: float = Field(1e-2, description="Weight decay to use.")
    attention: str = Field("xformers", description="Attention model.", choices=attentions)
    cache_latents: bool = Field(True, description="Cache latents.")
    center_crop: bool = Field(False,
                              description="[finetune, db_classic, controlnet] Whether to center crop the input images to the resolution.")
    checkpoints_total_limit: Optional[int] = Field(None,
                                                   description="[finetune, db_classic] Max number of checkpoints to store.")
    clip_skip: int = Field(1, description="[db] Clip skip.")
    cpu_only: bool = Field(False, description="[db] CPU only.")
    concepts_list: List[Dict] = Field([], description="[db] Concepts list.")
    concepts_path: str = Field("", description="[db] Path to the concepts.")
    controlnet_model_name_or_path: Optional[str] = Field(default=None, metadata={
        "help": "[controlnet] Path to pretrained controlnet model or model identifier from huggingface.co/models. If not specified controlnet weights are initialized from unet."})
    custom_model_name: str = Field("", description="[db] Custom model name.")
    deterministic: bool = Field(False, description="[db] Deterministic mode.")
    disable_class_matching: bool = Field(False, description="[db] Disable class matching.")
    disable_logging: bool = Field(False, description="[db] Disable logging.")
    ema_predict: bool = Field(False, description="[db] EMA predict.")
    epoch: int = Field(0, description="Lifetime trained epoch.")
    epoch_pause_frequency: int = Field(0, description="[db] Epoch pause frequency.")
    epoch_pause_time: int = Field(0, description="[db] Epoch pause time.")
    freeze_clip_normalization: bool = Field(False, description="Freeze clip normalization.")
    gradient_accumulation_steps: int = Field(default=1,
                                             description="Number of updates steps to accumulate before performing a backward/update pass.")
    gradient_checkpointing: bool = Field(False, description="Whether or not to use gradient checkpointing.")
    gradient_set_to_none: bool = Field(False, description="Whether or not to set gradients to None when zeroing.")
    graph_smoothing: float = Field(0.1, description="The scale of graph smoothing.")
    hflip: bool = Field(False, description="Horizontal flip.")
    infer_ema: bool = Field(False, description="Infer EMA.")  # REMOVE
    initial_revision: int = Field(0, description="Initial revision.")  # REMOVE
    input_pertubation: float = Field(0.1, description="[finetune] The scale of input pertubation. Recommended 0.1.")
    learning_rate: float = Field(5e-6, description="Initial learning rate.")
    learning_rate_min: float = Field(1e-6, description="Minimum learning rate.")
    lifetime_revision: int = Field(0, description="Lifetime revision.")
    local_rank: int = Field(-1, description="[finetune, controlnet] For distributed training: local_rank")  # ato
    lora_learning_rate: float = Field(1e-4, description="[db] LoRA learning rate.")
    lora_model_name: str = Field("", description="[db] LoRA model name.")
    lora_txt_learning_rate: float = Field(5e-5, description="[db] LoRA text learning rate.")
    lora_txt_rank: int = Field(4, description="[db] LoRA text rank.")
    lora_txt_weight: float = Field(1.0, description="[db] LoRA text weight.")
    lora_unet_rank: int = Field(4, description="[db] LoRA UNet rank.")
    lora_weight: float = Field(1.0, description="[db] LoRA weight.")
    lr_factor: float = Field(0.5, description="Learning rate factor.")  # ato
    lr_num_cycles: int = Field(1, description="Learning rate cycles.")  # ato
    lr_power: float = Field(1.0, description="Learning rate power.")  # ato
    lr_scale_pos: float = Field(0.5, description="Learning rate scale position.")  # ato
    lr_scheduler: str = Field("constant_with_warmup", description="Learning rate scheduler.")
    lr_warmup_steps: int = Field(500, description="Number of steps for the warmup in the lr scheduler.")
    max_grad_norm: float = Field(1.0, description="[finetune] Max gradient norm.")
    max_token_length: int = Field(75, description="[db] Max token length.")  # ato
    max_train_samples: Optional[int] = Field(default=None, description="[finetune, controlnet] For debugging purposes or quicker training, truncate the number of training examples to this value if set.")
    mixed_precision: Optional[str] = Field("no", description="Whether to use mixed precision.", choices=precisions)
    model_dir: str = Field("", description="Model directory.")
    model_name: str = Field("", description="Model name.")
    model_path: str = Field("", description="Model path.")
    noise_scheduler: str = Field("DDPM", description="[db] Noise scheduler during training?.")
    num_save_samples: int = Field(4, description="[finetune, controlnet] Number of samples to save.")
    num_train_epochs: int = Field(100, description="Number of training epochs.")
    offset_noise: float = Field(0, description="[finetune, db] The scale of noise offset.")
    optimizer: str = Field("8bit AdamW", description="Optimizer.")
    pad_tokens: bool = Field(True, description="[db] Pad tokens.")
    pretrained_model_name_or_path: str = Field("", description="Pretrained model name or path.")
    pretrained_vae_name_or_path: str = Field("", description="[db] Pretrained VAE model name or path.")  # ato
    prior_loss_scale: bool = Field(False, description="[db] Prior loss scale.")
    prior_loss_target: int = Field(100, description="[db] Prior loss target.")
    prior_loss_weight: float = Field(0.75, description="[db, db-classic] Prior loss weight.")
    prior_loss_weight_min: float = Field(0.1, description="[db] Minimum prior loss weight.")
    proportion_empty_prompts: float = Field(default=0,
                                            description="[controlnet] Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).")
    random_flip: bool = Field(False,
                              description="[finetune] Whether to randomly flip images horizontally.")  # Replace this with hflip, or replace hflip
    resolution: int = Field(512, description="Maximum resolution for input images.")
    revision: int = Field(0, description="Model Revision.")
    sample_batch_size: int = Field(1, description="[db] Sample batch size.")
    sanity_prompt: str = Field("", description="Sanity prompt.")  # Add to others?
    save_on_cancel: bool = Field(True, description="Save checkpoint when training is canceled.")
    save_embedding_every: int = Field(25, description="Save a checkpoint of the training state every X epochs.")
    save_lora_for_extra_net: bool = Field(True, description="Save LoRA for extra net.")
    save_preview_every: int = Field(5, description="Save preview every.")
    scale_lr: bool = Field(False, description="[finetune, db-classic] Scale the learning rate.")  # REMOVE
    scheduler: str = Field("ddim", description="Scheduler.")
    seed: int = Field(420420, description="Seed for reproducability, sanity prompt.")  # Add/standardize
    shuffle_tags: bool = Field(True, description="[db] Shuffle tags.")  # Add to others
    snapshot: Optional[str] = Field(None,
                                    description="Whether training should be resumed from a previous checkpoint. Use 'latest' to use the latest checkpoint in the output directory, or specify a revision.")
    snr_gamma: Optional[float] = Field(5.0,
                                       description="[finetune] SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0.")
    split_loss: bool = Field(True, description="Split loss.")
    src: str = Field("", description="The source checkpoint.")
    stop_text_encoder: float = Field(1.0, description="[db] Stop text encoder.")  # ato
    strict_tokens: bool = Field(False, description="[db] Strict tokens.")  # ato
    dynamic_img_norm: bool = Field(False, description="[db] Dynamic image normalization.")  # ato
    tenc_weight_decay: float = Field(0.01, description="[db] Text encoder weight decay.")
    tenc_grad_clip_norm: float = Field(0.00, description="[db] Text encoder gradient clipping norm.")
    tomesd: float = Field(0, description="[db] TomesD.")  # ato
    train_batch_size: int = Field(1, description="Batch size for the training dataloader.")
    train_data_dir: Optional[str] = Field(None,
                                          description="[finetune, controlnet, db-classic] A folder containing the training data.")
    train_mode: str = Field("db", description="The training mode to use.")
    train_unet: bool = Field(True, description="[db] Train UNet.")
    train_unfrozen: bool = Field(True, description="[db] Train unfrozen.")
    txt_learning_rate: float = Field(5e-6, description="[db] Text learning rate.")
    use_ema: bool = Field(False, description="[finetune, db] Whether to use EMA model.")
    use_dir_tags: bool = Field(False,
                               description="[finetune, controlnet, db-classic] Whether to use the directory name as the tag. Will be appended if not found in the caption.")
    train_lora: bool = Field(False, description="[db] Use LoRA.")
    use_lora_extended: bool = Field(False, description="[db] Use LoRA extended.")
    use_subdir: bool = Field(False, description="[db] Use subdirectory.")  # ato
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

            if not self.train_lora:
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
        if self.train_mode == "db" and self.concepts_path and required == -1:
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
        if self.train_lora:
            return self.src
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
            concepts_path_dir = Path(concepts_path).parent  # Get which folder is JSON file reside
            instance_data_dir = concept_data.get("instance_data_dir")
            if not os.path.isabs(instance_data_dir):
                print(f"Rebuilding portable concepts path: {concepts_path_dir} + {instance_data_dir}")
                concept_data["instance_data_dir"] = os.path.join(concepts_path_dir, instance_data_dir)

            concept = Concept(input_dict=concept_data)
            if concept.is_valid:
                concepts.append(concept.__dict__)
    except Exception as e:
        print(f"Exception parsing concepts: {e}")
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
        config = TrainingConfig(model_name=model_name)
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

    if model_dir:
        models_path = model_dir
        shared.dreambooth_models_path = models_path
    else:
        models_path = shared.dreambooth_models_path
        if models_path == "" or models_path is None:
            models_path = os.path.join(shared.models_path, "dreambooth")
    config_file = os.path.join(models_path, model_name, "train_config.json")
    try:
        with open(config_file, 'r') as openfile:
            config_dict = json.load(openfile)

        config = TrainingConfig(model_name=model_name)
        config.load_params(config_dict)
        shared.db_model_config = config
        return config
    except Exception as e:
        print(f"Exception loading config: {e}")
        traceback.print_exc()
        return None
