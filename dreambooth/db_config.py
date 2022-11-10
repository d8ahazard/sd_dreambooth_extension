import json
import os

from modules import paths, images


class DreamboothConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = None
        self.model_name = None
        self.scheduler = None
        self.src = None
        self.total_steps = None
        self.__dict__ = self

    def create_new(self, name, scheduler, src, total_steps):
        name = images.sanitize_filename_part(name, True)
        self.model_name = name
        self.scheduler = scheduler
        self.src = src
        self.total_steps = total_steps
        return self

    def from_ui(self,
                pretrained_model_name_or_path,
                instance_data_dir,
                class_data_dir,
                instance_prompt,
                use_filename_as_label,
                use_txt_as_label,
                class_prompt,
                save_sample_prompt,
                save_sample_negative_prompt,
                n_save_sample,
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
                hflip):

        pretrained_model_name_or_path = images.sanitize_filename_part(pretrained_model_name_or_path, True)
        models_path = paths.models_path
        model_dir = os.path.join(models_path, "dreambooth", pretrained_model_name_or_path)
        working_dir = os.path.join(model_dir, "working")
        with_prior_preservation = num_class_images > 0
        data = {"pretrained_model_name_or_path": pretrained_model_name_or_path,
                "instance_data_dir": instance_data_dir,
                "class_data_dir": class_data_dir,
                "instance_prompt": instance_prompt,
                "use_filename_as_label": use_filename_as_label,
                "use_txt_as_label": use_txt_as_label,
                "class_prompt": class_prompt,
                "save_sample_prompt": save_sample_prompt,
                "save_sample_negative_prompt": save_sample_negative_prompt,
                "n_save_sample": n_save_sample,
                "save_guidance_scale": save_guidance_scale,
                "save_infer_steps": save_infer_steps,
                "with_prior_preservation": with_prior_preservation,
                "num_class_images": num_class_images,
                "output_dir": model_dir,
                "working_dir": working_dir,
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
                "hflip": hflip,
                "prior_loss_weight": 1,
                "seed": None}
        for key in data:
            self.__dict__[key] = data[key]
        return self.__dict__

    def from_file(self, model_name):
        """
        Load config data from UI
        Args:
            model_name: The config to load

        Returns: Dict

        """
        model_name = images.sanitize_filename_part(model_name, True)
        model_path = paths.models_path
        config_file = os.path.join(model_path, "dreambooth", model_name, "db_config.json")
        try:
            with open(config_file, 'r') as openfile:
                config = json.load(openfile)
                for key in config:
                    self.__dict__[key] = config[key]
        except Exception as e:
            print(f"Exception loading config: {e}")
            return None
            pass
        return self.__dict__

    def save(self):
        """
        Save the config file3
        """
        model_path = paths.models_path
        config_file = os.path.join(model_path, "dreambooth", self.__dict__["model_name"], "db_config.json")
        with open(config_file, "w") as outfile:
            json.dump(self.__dict__, outfile)
