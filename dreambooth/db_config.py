import json
import os

from modules import paths, images, shared

try:
    cmd_dreambooth_models_path = shared.cmd_opts.dreambooth_models_path
except:
    cmd_dreambooth_models_path = None


class DreamboothConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = None
        self.model_name = None
        self.scheduler = None
        self.src = None
        self.total_steps = None
        self.revision = None
        self.instance_prompt = ""
        self.class_prompt = ""
        self.instance_token = ""
        self.class_token = ""
        self.instance_data_dir = ""
        self.class_data_dir = ""
        self.__dict__ = self

    def create_new(self, name, scheduler, src, total_steps):
        name = images.sanitize_filename_part(name, True)
        self.model_name = name
        self.scheduler = scheduler
        self.src = src
        self.total_steps = total_steps
        self.revision = total_steps
        return self

    def from_ui(self,
                model_dir,
                half_model,
                use_concepts,
                pretrained_vae_name_or_path,
                instance_data_dir,
                class_data_dir,
                instance_prompt,
                class_prompt,
                file_prompt_contents,
                instance_token,
                class_token,
                save_sample_prompt,
                save_sample_negative_prompt,
                n_save_sample,
                sample_seed,
                save_guidance_scale,
                save_infer_steps,
                num_class_images,
                resolution,
                center_crop,
                train_text_encoder,
                train_batch_size,
                sample_batch_size,
                num_train_epochs,
                max_train_steps,
                gradient_accumulation_steps,
                gradient_checkpointing,
                learning_rate,
                scale_lr,
                lr_scheduler,
                lr_warmup_steps,
                attention,
                use_8bit_adam,
                adam_beta1,
                adam_beta2,
                adam_weight_decay,
                adam_epsilon,
                max_grad_norm,
                save_preview_every,
                save_embedding_every,
                mixed_precision,
                not_cache_latents,
                concepts_list,
                use_cpu,
                pad_tokens,
                max_token_length,
                hflip,
                use_ema,
                class_negative_prompt,
                class_guidance_scale,
                class_infer_steps,
                shuffle_after_epoch
                ):

        model_dir = images.sanitize_filename_part(model_dir, True)
        dict = {}
        try:
            dict = self.from_file(model_dir)
        except:
            print(f"Exception loading model from path: {model_dir}")

        if "revision" not in self.__dict__:
            dict["revision"] = 0
        pretrained_vae_name_or_path = images.sanitize_filename_part(pretrained_vae_name_or_path, True)
        models_path = os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path
        model_dir = os.path.join(models_path, "dreambooth", model_dir)
        working_dir = os.path.join(model_dir, "working")
        with_prior_preservation = num_class_images is not None and num_class_images > 0
        dict["pretrained_model_name_or_path"] = working_dir

        data = {"pretrained_model_name_or_path": working_dir,
                "model_dir": model_dir,
                "half_model": half_model,
                "use_concepts": use_concepts,
                "pretrained_vae_name_or_path": pretrained_vae_name_or_path,
                "instance_data_dir": instance_data_dir,
                "class_data_dir": class_data_dir,
                "instance_prompt": instance_prompt,
                "class_prompt": class_prompt,
                "file_prompt_contents": file_prompt_contents,
                "instance_token": instance_token,
                "class_token": class_token,
                "save_sample_prompt": save_sample_prompt,
                "save_sample_negative_prompt": save_sample_negative_prompt,
                "n_save_sample": n_save_sample,
                "seed": sample_seed,
                "save_guidance_scale": save_guidance_scale,
                "save_infer_steps": save_infer_steps,
                "with_prior_preservation": with_prior_preservation,
                "num_class_images": num_class_images,
                "output_dir": model_dir,
                "resolution": resolution,
                "center_crop": center_crop,
                "train_text_encoder": train_text_encoder,
                "train_batch_size": train_batch_size,
                "sample_batch_size": sample_batch_size,
                "num_train_epochs": num_train_epochs,
                "max_train_steps": max_train_steps,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "gradient_checkpointing": gradient_checkpointing,
                "learning_rate": learning_rate,
                "scale_lr": scale_lr,
                "lr_scheduler": lr_scheduler,
                "lr_warmup_steps": lr_warmup_steps,
                "attention": attention,
                "use_8bit_adam": use_8bit_adam,
                "adam_beta1": adam_beta1,
                "adam_beta2": adam_beta2,
                "adam_weight_decay": adam_weight_decay,
                "adam_epsilon": adam_epsilon,
                "max_grad_norm": max_grad_norm,
                "save_preview_every": save_preview_every,
                "save_embedding_every": save_embedding_every,
                "mixed_precision": mixed_precision,
                "not_cache_latents": not_cache_latents,
                "concepts_list": concepts_list,
                "use_cpu": use_cpu,
                "pad_tokens": pad_tokens,
                "max_token_length": max_token_length,
                "hflip": hflip,
                "use_ema": use_ema,
                "prior_loss_weight": 1,
                "class_negative_prompt": class_negative_prompt,
                "class_guidance_scale": class_guidance_scale,
                "class_infer_steps": class_infer_steps,
                "shuffle_after_epoch": shuffle_after_epoch
        }
        for key in data:
            dict[key] = data[key]
        self.__dict__ = dict
        return self.__dict__

    def from_file(self, model_name):
        """
        Load config data from UI
        Args:
            model_name: The config to load

        Returns: Dict

        """
        model_name = images.sanitize_filename_part(model_name, True)
        model_path = os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path
        working_dir = os.path.join(model_path, "dreambooth", model_name, "working")
        config_file = os.path.join(model_path, "dreambooth", model_name, "db_config.json")
        try:
            with open(config_file, 'r') as openfile:
                config = json.load(openfile)
                if "max_token_length" not in config:
                    self.__dict__["max_token_length"] = 75
                    self.__dict__["model_dir"] = working_dir
                    self.__dict__["pretrained_model_name_or_path"] = working_dir
                if "class_negative_prompt" not in config:
                    self.__dict__["class_guidance_scale"] = 7.5
                    self.__dict__["class_negative_prompt"] = ""
                    self.__dict__["class_infer_steps"] = 60
                if "use_concepts" not in config:
                    self.__dict__["use_concepts"] = False
                if "half_model" not in config:
                    self.__dict__["half_model"] = False
                if "file_prompt_contents" not in config:
                    self.__dict__["file_prompt_contents"] = "description"
                    self.__dict__["instance_token"] = ""
                    self.__dict__["class_token"] = ""
                if "shuffle_after_epoch" not in config:
                    self.__dict__["shuffle_after_epoch"] = False

                for key in config:
                    self.__dict__[key] = config[key]
                if "revision" not in config:
                    if "total_steps" in config:
                        revision = config["total_steps"]
                    else:
                        revision = 0
                    self.__dict__["revision"] = revision
        except Exception as e:
            print(f"Exception loading config: {e}")
            return None
            pass
        return self.__dict__

    def save(self):
        """
        Save the config file
        """
        model_path = os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path
        config_file = os.path.join(model_path, "dreambooth", self.__dict__["model_name"], "db_config.json")
        with open(config_file, "w") as outfile:
            json.dump(self.__dict__, outfile, indent=4)
