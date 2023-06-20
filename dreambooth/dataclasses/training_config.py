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
from dreambooth.utils.utils import list_attention, list_precisions, list_schedulers, list_optimizer

# Keys to save, replacing our dumb __init__ method
save_keys = []

# Keys to return to the ui when Load Settings is clicked.
ui_keys = []


def sanitize_name(name):
    return "".join(x for x in name if (x.isalnum() or x in "._- "))


class TrainingConfig(BaseConfig):
    precisions = list_precisions()
    attentions = list_attention()
    optimizers = list_optimizer()
    schedulers = list_schedulers()

    # General
    num_train_epochs: int = Field(100, description="Number of training epochs.", title="Num Train Epochs", ge=1, le=10000, group="General")
    train_mode: str = Field("db", description="The training mode to use.", title="Train Mode",group="General")
    controlnet_model_name: str = Field("", description="[controlnet] Controlnet model name.",
                                       title="Controlnet Model Name", custom_type="controlnet_modelSelect")

    use_ema: bool = Field(False, description="[db, finetune] Whether to use EMA model.", title="Use EMA",group="General")
    train_lora: bool = Field(False, description="[db, finetune] Use LoRA.", title="Use LoRA",group="General")
    resolution: int = Field(512, description="Maximum resolution for input images.", title="Resolution", ge=8, multiple_of=8, le=4096, group="General")
    sample_batch_size: int = Field(1, description="Sample batch size.", title="Sample Batch Size", ge=1, le=1000, group="General")
    pretrained_vae_name_or_path: str = Field("", description="Pretrained VAE model name or path.", title="Custom VAE",
                                             custom_type="vae_modelSelect", group="General")
    noise_scheduler: str = Field("DDPM", description="Noise scheduler used during training.", title="Noise Scheduler",
                                 choices=["DDPM", "DDIM", "PNDM"], group="General")
    train_batch_size: int = Field(1, description="Batch size for the training dataloader.", title="Train Batch Size",
                                  gt=0, le=1000, group="General")

    gradient_accumulation_steps: int = Field(default=1,
                                             description="Number of updates steps to accumulate before performing a backward/update pass.",
                                             title="Gradient Accumulation Steps", ge=1, le=1000, group="General")

    # Training data
    concepts_list: List[Dict] = Field([], description="[db] Concepts list.", title="Concepts List", group="Training Data")
    concepts_path: str = Field("", description="[db] Path to the concepts.", title="Concepts Path", group="Training Data")
    disable_class_matching: bool = Field(False, description="[db] Disable class matching.",
                                         title="Disable Class Matching", group="Training Data")

    train_data_dir: Optional[str] = Field(None,
                                          description="[finetune, controlnet] A folder containing the training data.",
                                          title="Train Data Directory", group="Training Data")
    use_dir_tags: bool = Field(False,
                               description="[finetune, controlnet] Whether to use the directory name as the tag. Will be appended if not found in the caption.", title="Use Directory Tags", group="Training Data")


    # Performance
    attention: str = Field("xformers", description="Attention model.", choices=attentions, title="Attention", group="Performance")
    cache_latents: bool = Field(True, description="Cache latents.", title="Cache Latents", group="Performance")
    gradient_checkpointing: bool = Field(False, description="Whether or not to use gradient checkpointing.",
                                         title="Gradient Checkpointing", group="Performance")
    gradient_set_to_none: bool = Field(False, description="Whether or not to set gradients to None when zeroing.",
                                       title="Gradient Set To None", group="Performance")
    max_grad_norm: float = Field(1.0, description="Max gradient norm.", title="Max Gradient Normalization", ge=0, le=1, multiple_of=0.01, group="Performance")

    mixed_precision: Optional[str] = Field("no", description="Whether to use mixed precision.", choices=precisions, title="Mixed Precision", group="Performance")
    cpu_only: bool = Field(False, description="CPU only.", title="CPU Only", group="Performance")

    # Optimizer settings
    optimizer: str = Field("8bit AdamW", description="Optimizer.", title="Optimizer", choices=optimizers,
                           group="Optimizer")
    adam_beta1: float = Field(0.9, description="The beta1 parameter for the Adam optimizer.", title="Adam Beta 1", ge=0,
                              le=1, multiple_of=0.01, group="Optimizer")
    adam_beta2: float = Field(0.999, description="The beta2 parameter for the Adam optimizer.", title="Adam Beta 2",
                              ge=0, le=1, multiple_of=0.001, group="Optimizer")
    adam_epsilon: float = Field(1e-08, description="Epsilon value for the Adam optimizer.", title="Adam Epsilon",
                                group="Optimizer")
    adam_weight_decay: float = Field(1e-2, description="Weight decay to use.", title="Adam Weight Decay", ge=0, le=1,
                                     multiple_of=0.01, group="Optimizer")

    # Scheduler/LR
    learning_rate: float = Field(5e-6, description="Initial learning rate.", title="Learning Rate",
                                 group="Learning Rate")
    learning_rate_lora: float = Field(5e-5, description="[lora] LoRA learning rate.", title="LoRA Learning Rate",
                                      group="Learning Rate")
    learning_rate_lora_txt: float = Field(5e-5, description="[lora] LoRA txt learning rate.",
                                          title="LoRA Text Learning Rate", group="Learning Rate")
    learning_rate_min: float = Field(1e-6, description="Minimum learning rate.", title="Learning Rate Min",
                                     group="Learning Rate")
    learning_rate_txt: float = Field(5e-6, description="[db] Text learning rate.", title="Text Learning Rate",
                                     group="Learning Rate")
    lr_factor: float = Field(0.5, description="Learning rate factor.", title="LR Factor", ge=0, le=1, multiple_of=0.1,
                             group="Learning Rate")
    lr_num_cycles: int = Field(1, description="Learning rate cycles.", title="LR Cycle", ge=0, le=1, multiple_of=0.1,
                               group="Learning Rate")
    lr_power: float = Field(1.0, description="Learning rate power.", title="LR Power", ge=0, le=1, multiple_of=0.1,
                            group="Learning Rate")
    lr_scale_pos: float = Field(0.5, description="Learning rate scale position.", title="LR Scale position", ge=0, le=1,
                                multiple_of=0.1, group="Learning Rate")
    lr_scheduler: str = Field("constant_with_warmup", description="Learning rate scheduler.", title="LR Scheduler",
                              choices=schedulers, group="Learning Rate")
    lr_warmup_steps: int = Field(500, description="Number of steps for the warmup in the lr scheduler.",
                                 title="LR Warmup Steps", ge=0, le=10000, group="Learning Rate")

    # Preprocessing
    dynamic_img_norm: bool = Field(False, description="Dynamic image normalization.", title="Dynamic Image Normalization", group="Preprocessing")
    hflip: bool = Field(False, description="Horizontal flip.", title="Horizontal Flip", group="Preprocessing")
    input_pertubation: float = Field(0.1, description="The scale of input pertubation. Recommended 0.1.", title="Input Pertubation", ge=0, le=1, multiple_of=0.1, group="Preprocessing")
    offset_noise: float = Field(0, description="The scale of noise offset.", title="Offset Noise", ge=0, le=1, multiple_of=0.1, group="Preprocessing")
    max_token_length: int = Field(75, description="[db] Max token length.", title="Max Token Length", ge=75, le=1000,
                                  multiple_of=75, group="Preprocessing")
    pad_tokens: bool = Field(True, description="Pad tokens.", title="Pad Tokens", group="Preprocessing")
    shuffle_tags: bool = Field(True, description="Shuffle tags.", title="Shuffle Tags", group="Preprocessing")
    strict_tokens: bool = Field(False, description="Strict tokens.", title="Strict Tokens", group="Preprocessing")

    # Text encoder
    stop_text_encoder: float = Field(1.0, description="[db] Percentage of total training to train text encoder for.", title="Stop Text Pct", ge=0, le=1, multiple_of=0.01, group="Text Encoder")
    clip_skip: int = Field(1, description="[db] Number of CLIP Normalization layers to skip.", title="Clip Skip", ge=0, le=4, group="Text Encoder")
    freeze_clip_normalization: bool = Field(False, description="Freeze clip normalization.",title="Freeze Clip Normalization", group="Text Encoder")
    tenc_weight_decay: float = Field(0.01, description="[db] Text encoder weight decay.", title="Tenc Weight Decay", ge=0, le=1, multiple_of=0.01, group="Text Encoder")
    tenc_grad_clip_norm: float = Field(0.00, description="[db] Text encoder gradient clipping norm.", title="Tenc Grad Clip Norm", ge=0, le=1, multiple_of=0.01, group="Text Encoder")
    train_unfrozen: bool = Field(True, description="[db] Train unfrozen.",  title="Train Unfrozen", group="Text Encoder")

    # LORA
    lora_model_name: str = Field("", description="[lora] LoRA model name.", title="LoRA Model Name", custom_type="loras_modelSelect", group="LoRA")
    lora_txt_rank: int = Field(4, description="[lora] LoRA text rank.", title="LoRA Text Rank", ge=2, le=16, group="LoRA")
    lora_txt_weight: float = Field(1.0, description="[lora] LoRA text weight.", title="LoRA Text Weight", ge=-3, le=3, multiple_of=0.1, group="LoRA")
    lora_unet_rank: int = Field(4, description="[lora] LoRA UNet rank.", title="LoRA UNet Rank", ge=2, le=16, group="LoRA")
    lora_weight: float = Field(1.0, description="[lora] LoRA weight.", title="LoRA Weight", ge=-3, le=3, multiple_of=0.1, group="LoRA")

    # Dreambooth
    prior_loss_scale: bool = Field(False, description="[db] Prior loss scale.", title="Prior Loss Scale",group="Dreambooth")
    prior_loss_target: int = Field(100, description="[db] Prior loss target.", title="Prior Loss Target", ge=0, le=1000, group="Dreambooth")
    prior_loss_weight: float = Field(0.75, description="[db] Prior loss weight.", title="Prior Loss Weight", ge=0, le=1, multiple_of=0.1, group="Dreambooth")
    prior_loss_weight_min: float = Field(0.1, description="[db] Minimum prior loss weight.", title="Prior Loss Minimum", ge=0, le=1, multiple_of=0.1, group="Dreambooth")
    proportion_empty_prompts: float = Field(default=0, description="[controlnet] Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).", title="Pct Empty Prompts", ge=0, le=1, multiple_of=0.1, group="Dreambooth")
    split_loss: bool = Field(True, description="[db] Split loss.", title="Split Loss", group="Dreambooth")
    train_unet: bool = Field(True, description="[db] Train UNet.", title="Train UNet", group="Dreambooth")

    # Model data, these aren't shown directly in the UI
    epoch: int = Field(0, description="[model] Lifetime trained epoch.", title="Epoch")
    lifetime_revision: int = Field(0, description="[model] Lifetime revision.", title="Lifetime Revision")
    model_dir: str = Field("", description="[model] Model directory.", title="Model Directory")
    model_name: str = Field("", description="[model] Model name.", title="Model Name")
    model_path: str = Field("", description="[model] Model path.", title="Model Path")
    pretrained_model_name_or_path: str = Field("", description="[model] Pretrained model name or path.", title="Pretrained Model Name or Path")

    revision: int = Field(0, description="[model] Model Revision.", title="Revision")
    scheduler: str = Field("ddim", description="[model] Scheduler.", title="Scheduler",
                           choices=["ddim", "ddpm", "pndm"])
    src: str = Field("", description="[model] The source checkpoint.", title="Source Checkpoint")
    v2: bool = Field(False, description="[model] If this is a V2 Model or not.", title="V2")

    # Samples, saving
    checkpoint: Optional[str] = Field(None,
                                      description="Whether training should be resumed from a previous checkpoint. Use 'latest' to use the latest checkpoint in the output directory, or specify a revision.",
                                      title="Snapshot", custom_type="snapshot_modelSelect", group="Saving")
    checkpoints_total_limit: Optional[int] = Field(0, description="[finetune] Max number of checkpoints to store.",
                                                   title="Checkpoints Total Limit", ge=0, le=100, group="Saving")
    max_train_samples: Optional[int] = Field(default=None,
                                             description="[finetune, controlnet] For debugging purposes or quicker training, truncate the number of training examples to this value if set.",
                                             title="Max Train Samples", ge=0, le=10000, group="Saving")
    disable_logging: bool = Field(False, description="Disable log parsing.", title="Disable Logging", group="Saving")
    graph_smoothing: float = Field(0.1, description="The scale of graph smoothing.", title="Graph Smoothing", ge=0,
                                   le=1, multiple_of=0.1, group="Saving")
    num_save_samples: int = Field(4, description="[finetune, controlnet] Number of samples to save.", title="Num Save Samples", ge=0, le=1000, group="Saving")
    sanity_prompt: str = Field("", description="Sanity prompt.", title="Sanity Prompt", group="Saving")
    save_on_cancel: bool = Field(True, description="Save checkpoint when training is canceled.", title="Save on Cancel", group="Saving")
    save_embedding_every: int = Field(25, description="Save a checkpoint of the training state every X epochs.", title="Save Weights Frequency", ge=0, le=1000, group="Saving")
    save_preview_every: int = Field(5, description="Save preview every.", title="Save Preview Frequency", ge=0, le=1000, group="Saving")
    seed: int = Field(420420, description="Seed for reproducibility, sanity prompt.", title="Seed", ge=-1, le=21474836147, group="Saving")
    simulate_training: bool = Field(False, description="Simulate training.", title="Simulate Training", group="Saving")
    snr_gamma: Optional[float] = Field(5.0, description="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0.", title="SNR Gamma", le=0, ge=10, multiple_of=0.1, group="Saving")
    tomesd: bool = Field(True, description="Apply TomeSD when generating images.", title="Use TomeSD", group="Saving")


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
