import json
import os
import traceback

from extensions.sd_dreambooth_extension.dreambooth.db_concept import Concept
from extensions.sd_dreambooth_extension.dreambooth import db_shared as shared


def sanitize_name(name):
    return "".join(x for x in name if (x.isalnum() or x in "._- "))


class DreamboothConfig:
    v2 = False
    save_class_txt = False
    scheduler = "ddim"
    lifetime_revision = 0
    initial_revision = 0
    epoch = 0
    resolution = 512

    def __init__(self,
                 model_name: str = "",
                 attention: str = "default",
                 cache_latents=True,
                 center_crop: bool = True,
                 clip_skip: int = 1,
                 concepts_path: str = "",
                 custom_model_name: str = "",
                 epoch: int = 0,
                 epoch_pause_frequency: int = 0,
                 epoch_pause_time: int = 0,
                 gradient_accumulation_steps: int = 1,
                 gradient_checkpointing: bool = True,
                 gradient_set_to_none: bool = True,
                 graph_smoothing: int = 50,
                 half_model: bool = False,
                 has_ema: bool = False,
                 hflip: bool = False,
                 learning_rate: float = 5e-6,
                 learning_rate_min: float = 1e-6,
                 lora_learning_rate: float = 1e-4,
                 lora_txt_learning_rate: float = 5e-5,
                 lora_txt_weight: float = 1.0,
                 lora_weight: float = 1.0,
                 lr_cycles: int = 1,
                 lr_factor: float = 0.5,
                 lr_power: float = 1.0,
                 lr_scale_pos: float = 0.5,
                 lr_scheduler: str = 'constant',
                 lr_warmup_steps: int = 0,
                 max_token_length: int = 75,
                 mixed_precision: str = "fp16",
                 model_path: str = "",
                 num_train_epochs: int = 100,
                 pad_tokens: bool = True,
                 pretrained_vae_name_or_path: str = "",
                 prior_loss_weight: float = 1.0,
                 resolution: int = 512,
                 revision: int = 0,
                 sample_batch_size: int = 1,
                 sanity_prompt: str = "",
                 sanity_seed: int = 420420,
                 save_ckpt_after: bool = True,
                 save_ckpt_cancel: bool = False,
                 save_ckpt_during: bool = True,
                 save_embedding_every: int = 25,
                 save_lora_after: bool = True,
                 save_lora_cancel: bool = False,
                 save_lora_during: bool = True,
                 save_preview_every: int = 5,
                 save_state_after: bool = False,
                 save_state_cancel: bool = False,
                 save_state_during: bool = False,
                 scheduler: str = "ddim",
                 src: str = "",
                 shuffle_tags: bool = False,
                 train_batch_size: int = 1,
                 stop_text_encoder: int = 0,
                 use_8bit_adam: bool = True,
                 use_concepts: bool = False,
                 use_ema: bool = True,
                 use_lora: bool = False,
                 v2: bool = False,
                 c1_class_data_dir: str = "",
                 c1_class_guidance_scale: float = 7.5,
                 c1_class_infer_steps: int = 60,
                 c1_class_negative_prompt: str = "",
                 c1_class_prompt: str = "",
                 c1_class_token: str = "",
                 c1_instance_data_dir: str = "",
                 c1_instance_prompt: str = "",
                 c1_instance_token: str = "",
                 c1_n_save_sample: int = 1,
                 c1_num_class_images: int = 0,
                 c1_sample_seed: int = -1,
                 c1_save_guidance_scale: float = 7.5,
                 c1_save_infer_steps: int = 60,
                 c1_save_sample_negative_prompt: str = "",
                 c1_save_sample_prompt: str = "",
                 c1_save_sample_template: str = "",
                 c2_class_data_dir: str = "",
                 c2_class_guidance_scale: float = 7.5,
                 c2_class_infer_steps: int = 60,
                 c2_class_negative_prompt: str = "",
                 c2_class_prompt: str = "",
                 c2_class_token: str = "",
                 c2_instance_data_dir: str = "",
                 c2_instance_prompt: str = "",
                 c2_instance_token: str = "",
                 c2_n_save_sample: int = 1,
                 c2_num_class_images: int = 0,
                 c2_sample_seed: int = -1,
                 c2_save_guidance_scale: float = 7.5,
                 c2_save_infer_steps: int = 60,
                 c2_save_sample_negative_prompt: str = "",
                 c2_save_sample_prompt: str = "",
                 c2_save_sample_template: str = "",
                 c3_class_data_dir: str = "",
                 c3_class_guidance_scale: float = 7.5,
                 c3_class_infer_steps: int = 60,
                 c3_class_negative_prompt: str = "",
                 c3_class_prompt: str = "",
                 c3_class_token: str = "",
                 c3_instance_data_dir: str = "",
                 c3_instance_prompt: str = "",
                 c3_instance_token: str = "",
                 c3_n_save_sample: int = 1,
                 c3_num_class_images: int = 0,
                 c3_sample_seed: int = -1,
                 c3_save_guidance_scale: float = 7.5,
                 c3_save_infer_steps: int = 60,
                 c3_save_sample_negative_prompt: str = "",
                 c3_save_sample_prompt: str = "",
                 c3_save_sample_template: str = "",
                 concepts_list=None,
                 **kwargs
                 ):
        if revision == "" or revision is None:
            revision = 0
        if epoch == "" or epoch is None:
            epoch = 0
        model_name = "".join(x for x in model_name if (x.isalnum() or x in "._- "))
        models_path = shared.dreambooth_models_path
        if models_path == "" or models_path is None:
            models_path = os.path.join(shared.models_path, "dreambooth")
        model_dir = os.path.join(models_path, model_name)
        working_dir = os.path.join(model_dir, "working")
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        self.attention = attention
        self.cache_latents = cache_latents
        self.center_crop = center_crop
        self.clip_skip = clip_skip
        self.concepts_path = concepts_path
        self.custom_model_name = custom_model_name
        self.epoch = epoch
        self.epoch_pause_frequency = epoch_pause_frequency
        self.epoch_pause_time = epoch_pause_time
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_set_to_none = gradient_set_to_none
        self.graph_smoothing = graph_smoothing
        self.half_model = half_model
        self.hflip = hflip
        self.learning_rate = learning_rate
        self.learning_rate_min = learning_rate_min
        self.lora_learning_rate = lora_learning_rate
        self.lora_txt_learning_rate = lora_txt_learning_rate
        self.lora_txt_weight = lora_txt_weight
        self.lora_weight = lora_weight
        self.lr_cycles = lr_cycles
        self.lr_factor = lr_factor
        self.lr_power = lr_power
        self.lr_scale_pos = lr_scale_pos
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.max_token_length = max_token_length
        self.mixed_precision = mixed_precision
        self.model_dir = model_dir
        self.model_name = model_name
        self.num_train_epochs = num_train_epochs
        self.pad_tokens = pad_tokens
        self.pretrained_model_name_or_path = working_dir
        self.pretrained_vae_name_or_path = pretrained_vae_name_or_path
        self.prior_loss_weight = prior_loss_weight
        self.resolution = resolution
        self.revision = int(revision)
        self.sample_batch_size = sample_batch_size
        self.sanity_prompt = sanity_prompt
        self.sanity_seed = sanity_seed
        self.save_ckpt_after = save_ckpt_after
        self.save_ckpt_cancel = save_ckpt_cancel
        self.save_ckpt_during = save_ckpt_during
        self.save_embedding_every = save_embedding_every
        self.save_lora_after = save_lora_after
        self.save_lora_cancel = save_lora_cancel
        self.save_lora_during = save_lora_during
        self.save_preview_every = save_preview_every
        self.save_state_after = save_state_after
        self.save_state_cancel = save_state_cancel
        self.save_state_during = save_state_during
        self.src = src
        self.shuffle_tags = shuffle_tags
        self.train_batch_size = train_batch_size
        self.stop_text_encoder = stop_text_encoder
        self.use_8bit_adam = use_8bit_adam
        self.use_concepts = use_concepts
        self.use_ema = False if use_ema is None else use_ema
        self.use_lora = False if use_lora is None else use_lora
        if scheduler is not None:
            self.scheduler = scheduler

        if v2 == 'True':
            self.v2 = True
        elif v2 == 'False':
            self.v2 = False
        else:
            self.v2 = v2

        self.has_ema = has_ema

        if concepts_list is None:
            if self.use_concepts:
                if concepts_path is not None and concepts_path != "" and os.path.exists(concepts_path):
                    try:
                        self.concepts_list = []
                        with open(concepts_path, "r") as f:
                            concepts = json.load(f)
                            for concept in concepts:
                                self.concepts_list.append(Concept(input_dict=concept))
                    except:
                        print("Exception opening concepts JSON.")
                        traceback.print_exc()
                else:
                    print("Please provide a valid concepts path.")
            else:
                concept1 = Concept(c1_instance_data_dir, c1_class_data_dir,
                                   c1_instance_prompt,
                                   c1_class_prompt, c1_save_sample_prompt, c1_save_sample_template, c1_instance_token,
                                   c1_class_token,
                                   c1_num_class_images, c1_class_negative_prompt, c1_class_guidance_scale,
                                   c1_class_infer_steps,
                                   c1_save_sample_negative_prompt, c1_n_save_sample, c1_sample_seed,
                                   c1_save_guidance_scale,
                                   c1_save_infer_steps)

                concept2 = Concept(c2_instance_data_dir, c2_class_data_dir,
                                   c2_instance_prompt,
                                   c2_class_prompt, c2_save_sample_prompt, c2_save_sample_template, c2_instance_token,
                                   c2_class_token,
                                   c2_num_class_images, c2_class_negative_prompt, c2_class_guidance_scale,
                                   c2_class_infer_steps,
                                   c2_save_sample_negative_prompt, c2_n_save_sample, c2_sample_seed,
                                   c2_save_guidance_scale,
                                   c2_save_infer_steps)

                concept3 = Concept(c3_instance_data_dir, c3_class_data_dir,
                                   c3_instance_prompt,
                                   c3_class_prompt, c3_save_sample_prompt, c3_save_sample_template, c3_instance_token,
                                   c3_class_token,
                                   c3_num_class_images, c3_class_negative_prompt, c3_class_guidance_scale,
                                   c3_class_infer_steps,
                                   c3_save_sample_negative_prompt, c3_n_save_sample, c3_sample_seed,
                                   c3_save_guidance_scale,
                                   c3_save_infer_steps)

                concepts = [concept1, concept2, concept3]
                self.concepts_list = []
                c_count = 0
                for concept in concepts:
                    if concept.is_valid():
                        if concept.class_data_dir == "" or concept.class_data_dir is None or concept.class_data_dir == shared.script_path:
                            class_dir = os.path.join(model_dir, f"classifiers_{c_count}")
                            if not os.path.exists(class_dir):
                                os.makedirs(class_dir)
                            concept.class_data_dir = class_dir
                        self.concepts_list.append(concept)
                    c_count += 1
        else:
            if len(concepts_list):
                self.concepts_list = concepts_list

    def save(self):
        """
        Save the config file
        """
        self.lifetime_revision = self.initial_revision + self.revision
        models_path = shared.dreambooth_models_path
        if models_path == "" or models_path is None:
            models_path = os.path.join(shared.models_path, "dreambooth")

        config_file = os.path.join(models_path, self.model_name, "db_config.json")
        with open(config_file, "w") as outfile:
            json.dump(self.__dict__, outfile, indent=4)


def save_json(model_name: str, json_cfg: str):
    """
    Save the config file
    """
    models_path = shared.dreambooth_models_path
    if models_path == "" or models_path is None:
        models_path = os.path.join(shared.models_path, "dreambooth")
    if not os.path.exists(os.path.join(models_path, model_name)):
        os.makedirs(os.path.join(models_path, model_name))
    config_file = os.path.join(models_path, model_name, "db_config.json")
    try:
        loaded = json.loads(json_cfg)
        with open(config_file, "w") as outfile:
            json.dump(loaded, outfile, indent=4)

        return json.dumps(loaded)
    except Exception as e:
        traceback.print_exc()
        return {"exception": f"{e}"}


def save_config(*args):
    config = DreamboothConfig(*args)
    print("Saved settings.")
    config.save()
    #return f"Saved settings for model: {config.model_name}"


def from_file(model_name):
    """
    Load config data from UI
    Args:
        model_name: The config to load

    Returns: Dict

    """
    if model_name == "" or model_name is None:
        return None
    if isinstance(model_name, list):
        model_name = model_name[0]
    model_name = sanitize_name(model_name)
    models_path = shared.dreambooth_models_path
    if models_path == "" or models_path is None:
        models_path = os.path.join(shared.models_path, "dreambooth")
    working_dir = os.path.join(models_path, model_name, "working")
    config_file = os.path.join(models_path, model_name, "db_config.json")
    try:
        with open(config_file, 'r') as openfile:
            config_dict = json.load(openfile)
            if config_dict is None:
                print("Invalid config dict.")
                return config_dict
            concept_keys = ["instance_data_dir", "class_data_dir", "instance_prompt",
                            "class_prompt", "save_sample_prompt", "save_sample_template", "instance_token",
                            "class_token", "num_class_images",
                            "class_negative_prompt", "class_guidance_scale", "class_infer_steps",
                            "save_sample_negative_prompt", "n_save_sample", "sample_seed", "save_guidance_scale",
                            "save_infer_steps"]
            skip_keys = ["seed", "shuffle_after_epoch", "total_steps", "output_dir", "with_prior_preservation",
                         "negative_prompt", "file_prompt_contents"]
            concept_dict = {}
            has_old_concept = False
            # Ensure we aren't using any old keys
            if "not_cache_latents" in config_dict:
                config_dict["cache_latents"] = not config_dict["not_cache_latents"]
                config_dict.pop("not_cache_latents")
            for skip_key in skip_keys:
                if skip_key in config_dict:
                    config_dict.pop(skip_key)
            for concept_key in concept_keys:
                if concept_key in config_dict:
                    has_old_concept = True
                    concept_dict[concept_key] = config_dict.pop(concept_key)
            config_dict["pretrained_model_name_or_path"] = working_dir
            config = DreamboothConfig(**config_dict)

            if not config.use_concepts and has_old_concept:
                concept = Concept(input_dict=concept_dict)

            concepts = []
            if "concepts_list" in config.__dict__:
                if not len(config.concepts_list) and has_old_concept:
                    concepts.append(concept)
                else:
                    for concept_dict in config.concepts_list:
                        concept = Concept(input_dict=concept_dict)
                        concepts.append(concept)
            c_idx = 0
            for concept in concepts:
                if "class_data_dir" not in concept.__dict__ or concept.class_data_dir == "" \
                        or concept.class_data_dir is None:
                    concept.class_data_dir = os.path.join(config.model_dir, f"classifiers_{c_idx}")
                    c_idx += 1
            config.concepts_list = concepts
            if config.revision == "" or config.revision is None:
                config.revision = 0
            else:
                config.revision = int(config.revision)
            if config.epoch == "" or config.epoch is None:
                config.epoch = 0
            else:
                config.epoch = int(config.epoch)
            return config
    except Exception as e:
        print(f"Exception loading config: {e}")
        traceback.print_exc()
        return None
