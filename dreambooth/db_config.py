import json
import os
import traceback

from modules import paths, images, shared

try:
    cmd_dreambooth_models_path = shared.cmd_opts.dreambooth_models_path
except:
    cmd_dreambooth_models_path = None


class DreamboothConfig:
    def __init__(self, name=None, scheduler="ddim", src=None, total_steps=0):
        if name is not None:
            name = images.sanitize_filename_part(name, True)
            self.model_name = name
            models_path = shared.models_path
            models_path = os.path.join(models_path, "dreambooth")
            if shared.cmd_opts.dreambooth_models_path is not None:
                models_path = shared.cmd_opts.dreambooth_models_path
            self.model_dir = os.path.join(models_path, name)
            self.pretrained_model_name_or_path = os.path.join(self.model_dir, "working")
            if not os.path.exists(self.pretrained_model_name_or_path):
                os.makedirs(self.pretrained_model_name_or_path)
        else:
            self.model_dir = None
            self.pretrained_model_name_or_path = None

        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.adam_weight_decay = 0.01
        self.attention = "default"
        self.center_crop = True
        self.class_data_dir = None
        self.class_guidance_scale = 7.5
        self.class_infer_steps = 60
        self.class_negative_prompt = None
        self.class_prompt = ""
        self.class_token = None
        self.concepts_list = None
        self.concepts_path = None
        self.file_prompt_contents = "Description"
        self.gradient_accumulation_steps = 1
        self.gradient_checkpointing = True
        self.half_model = False
        self.hflip = False
        self.instance_data_dir = None
        self.instance_prompt = None
        self.instance_token = None
        self.learning_rate = 0.000002
        self.lr_scheduler = "constant"
        self.lr_warmup_steps = 0
        self.max_grad_norm = 1
        self.max_token_length = 75
        self.max_train_steps = 1000
        self.mixed_precision = "fp16"
        self.n_save_sample = 4
        self.not_cache_latents = True
        self.num_class_images = 0
        self.num_train_epochs = 1
        self.output_dir = self.model_dir
        self.pad_tokens = True
        self.pretrained_vae_name_or_path = None
        self.prior_loss_weight = 1
        self.resolution = 512
        self.revision = total_steps
        self.sample_batch_size = 1
        self.sample_seed = -1
        self.save_embedding_every = 500
        self.save_guidance_scale = 7.5
        self.save_infer_steps = 60
        self.save_preview_every = 500
        self.save_sample_negative_prompt = None
        self.save_sample_prompt = None
        self.scale_lr = False
        self.scheduler = scheduler
        self.seed = -1
        self.shuffle_after_epoch = False
        self.src = src
        self.total_steps = total_steps
        self.train_batch_size = 1
        self.train_text_encoder = False
        self.use_8bit_adam = False
        self.use_concepts = False
        self.use_cpu = False
        self.use_ema = True
        self.with_prior_preservation = False

    def save_params(self,
                    model_dir, max_train_steps, num_train_epochs, save_embedding_every,
                    save_preview_every,
                    learning_rate, scale_lr, lr_scheduler, lr_warmup_steps, resolution, center_crop,
                    hflip, pretrained_vae_name_or_path, use_concepts, concepts_path,
                    train_batch_size, sample_batch_size, use_cpu, use_8bit_adam, mixed_precision,
                    attention, train_text_encoder, use_ema, pad_tokens, max_token_length,
                    gradient_checkpointing, gradient_accumulation_steps, max_grad_norm, adam_beta1,
                    adam_beta2, adam_weight_decay, adam_epsilon, c1_max_steps, c1_instance_data_dir,
                    c1_class_data_dir,
                    c1_file_prompt_contents, c1_instance_prompt, c1_class_prompt, c1_save_sample_prompt,
                    c1_instance_token,
                    c1_class_token, c1_num_class_images, c1_class_negative_prompt, c1_class_guidance_scale,
                    c1_class_infer_steps, c1_save_sample_negative_prompt, c1_n_save_sample, c1_sample_seed,
                    c1_save_guidance_scale, c1_save_infer_steps, c2_max_steps, c2_instance_data_dir, c2_class_data_dir,
                    c2_file_prompt_contents, c2_instance_prompt, c2_class_prompt, c2_save_sample_prompt,
                    c2_instance_token, c2_class_token, c2_num_class_images, c2_class_negative_prompt,
                    c2_class_guidance_scale, c2_class_infer_steps, c2_save_sample_negative_prompt,
                    c2_n_save_sample, c2_sample_seed, c2_save_guidance_scale, c2_save_infer_steps, c3_max_steps,
                    c3_instance_data_dir, c3_class_data_dir, c3_file_prompt_contents, c3_instance_prompt,
                    c3_class_prompt, c3_save_sample_prompt, c3_instance_token, c3_class_token, c3_num_class_images,
                    c3_class_negative_prompt, c3_class_guidance_scale, c3_class_infer_steps,
                    c3_save_sample_negative_prompt,
                    c3_n_save_sample, c3_sample_seed, c3_save_guidance_scale, c3_save_infer_steps
                    ):
        model_name = images.sanitize_filename_part(model_dir, True)
        pretrained_vae_name_or_path = images.sanitize_filename_part(pretrained_vae_name_or_path, True)
        models_path = os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path
        model_dir = os.path.join(models_path, "dreambooth", model_name)
        working_dir = os.path.join(model_dir, "working")
        self.model_name = model_name
        self.pretrained_model_name_or_path = working_dir
        self.model_dir = model_dir
        self.max_train_steps = max_train_steps
        self.num_train_epochs = num_train_epochs
        self.save_embedding_every = save_embedding_every
        self.save_preview_every = save_preview_every
        self.learning_rate = learning_rate
        self.scale_lr = scale_lr
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.resolution = resolution
        self.center_crop = center_crop
        self.hflip = hflip
        self.pretrained_vae_name_or_path = pretrained_vae_name_or_path
        self.use_concepts = use_concepts
        self.concepts_path = concepts_path
        self.train_batch_size = train_batch_size
        self.sample_batch_size = sample_batch_size
        self.use_cpu = use_cpu
        self.use_8bit_adam = use_8bit_adam
        self.mixed_precision = mixed_precision
        self.attention = attention
        self.train_text_encoder = train_text_encoder
        self.use_ema = use_ema
        self.pad_tokens = pad_tokens
        self.max_token_length = max_token_length
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_weight_decay = adam_weight_decay
        self.adam_epsilon = adam_epsilon

        if self.use_concepts:
            if concepts_path is not None and concepts_path != "" and os.path.exists(concepts_path):
                try:
                    with open(concepts_path, "r") as f:
                        self.concepts_list = json.load(f)
                except:
                    print("Exception opening concepts JSON.")
                    traceback.print_exc()
            else:
                print("Please provide a valid concepts path.")
        else:
            concept1 = Concept(c1_max_steps, c1_instance_data_dir, c1_class_data_dir, c1_file_prompt_contents,
                               c1_instance_prompt,
                               c1_class_prompt, c1_save_sample_prompt, c1_instance_token, c1_class_token,
                               c1_num_class_images, c1_class_negative_prompt, c1_class_guidance_scale,
                               c1_class_infer_steps,
                               c1_save_sample_negative_prompt, c1_n_save_sample, c1_sample_seed, c1_save_guidance_scale,
                               c1_save_infer_steps)

            concept2 = Concept(c2_max_steps, c2_instance_data_dir, c2_class_data_dir, c2_file_prompt_contents,
                               c2_instance_prompt,
                               c2_class_prompt, c2_save_sample_prompt, c2_instance_token, c2_class_token,
                               c2_num_class_images, c2_class_negative_prompt, c2_class_guidance_scale,
                               c2_class_infer_steps,
                               c2_save_sample_negative_prompt, c2_n_save_sample, c2_sample_seed, c2_save_guidance_scale,
                               c2_save_infer_steps)

            concept3 = Concept(c3_max_steps, c3_instance_data_dir, c3_class_data_dir, c3_file_prompt_contents,
                               c3_instance_prompt,
                               c3_class_prompt, c3_save_sample_prompt, c3_instance_token, c3_class_token,
                               c3_num_class_images, c3_class_negative_prompt, c3_class_guidance_scale,
                               c3_class_infer_steps,
                               c3_save_sample_negative_prompt, c3_n_save_sample, c3_sample_seed, c3_save_guidance_scale,
                               c3_save_infer_steps)

            self.concepts_list = [concept1, concept2, concept3]
        self.save()

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
        base_dict = {}
        try:
            base_dict = self.from_file(model_dir).__dict__
        except:
            print(f"Exception loading model from path: {model_dir}")

        pretrained_vae_name_or_path = images.sanitize_filename_part(pretrained_vae_name_or_path, True)
        models_path = os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path
        model_dir = os.path.join(models_path, "dreambooth", model_dir)
        working_dir = os.path.join(model_dir, "working")
        with_prior_preservation = num_class_images is not None and num_class_images > 0
        base_dict["pretrained_model_name_or_path"] = working_dir
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.adam_weight_decay = adam_weight_decay
        self.attention = attention
        self.center_crop = center_crop
        self.class_data_dir = class_data_dir
        self.class_guidance_scale = class_guidance_scale
        self.class_infer_steps = class_infer_steps
        self.class_negative_prompt = class_negative_prompt
        self.class_prompt = class_prompt
        self.class_token = class_token
        self.concepts_list = concepts_list
        self.file_prompt_contents = file_prompt_contents
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.half_model = half_model
        self.hflip = hflip
        self.instance_data_dir = instance_data_dir
        self.instance_prompt = instance_prompt
        self.instance_token = instance_token
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.max_grad_norm = max_grad_norm
        self.max_token_length = max_token_length
        self.max_train_steps = max_train_steps
        self.mixed_precision = mixed_precision
        self.model_dir = model_dir
        self.n_save_sample = n_save_sample
        self.not_cache_latents = not_cache_latents
        self.num_class_images = num_class_images
        self.num_train_epochs = num_train_epochs
        self.output_dir = model_dir
        self.pad_tokens = pad_tokens
        self.pretrained_model_name_or_path = working_dir
        self.pretrained_vae_name_or_path = pretrained_vae_name_or_path
        self.prior_loss_weight = 1
        self.resolution = resolution
        self.sample_batch_size = sample_batch_size
        self.save_embedding_every = save_embedding_every
        self.save_guidance_scale = save_guidance_scale
        self.save_infer_steps = save_infer_steps
        self.save_preview_every = save_preview_every
        self.save_sample_negative_prompt = save_sample_negative_prompt
        self.save_sample_prompt = save_sample_prompt
        self.scale_lr = scale_lr
        self.seed = sample_seed
        self.shuffle_after_epoch = shuffle_after_epoch
        self.train_batch_size = train_batch_size
        self.train_text_encoder = train_text_encoder
        self.use_8bit_adam = use_8bit_adam
        self.use_concepts = use_concepts
        self.use_cpu = use_cpu
        self.use_ema = use_ema
        self.with_prior_preservation = with_prior_preservation

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
        except Exception as e:
            print(f"Exception loading config: {e}")
            return None
            pass
        self.__dict__ = config
        return self

    def save(self):
        """
        Save the config file
        """
        model_path = os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path
        config_file = os.path.join(model_path, "dreambooth", self.model_name, "db_config.json")
        with open(config_file, "w") as outfile:
            json.dump(self.__dict__, outfile, indent=4)


class Concept(dict):
    def __init__(self, max_steps: int, instance_data_dir: str, class_data_dir: str, file_prompt_contents: str,
                 instance_prompt: str, class_prompt: str, save_sample_prompt: str, instance_token: str,
                 class_token: str, num_class_image: int, class_negative_prompt: str, class_guidance_scale: float,
                 class_infer_steps: int, save_sample_negative_prompt: str, n_save_sample: int, sample_seed: int,
                 save_guidance_scale: float, save_infer_steps: int):
        self.max_steps = max_steps
        self.instance_data_dir = instance_data_dir
        self.class_data_dir = class_data_dir
        self.file_prompt_contents = file_prompt_contents
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.save_sample_prompt = save_sample_prompt
        self.instance_token = instance_token
        self.class_token = class_token
        self.num_class_image = num_class_image
        self.class_negative_prompt = class_negative_prompt
        self.class_guidance_scale = class_guidance_scale
        self.class_infer_steps = class_infer_steps
        self.save_sample_negative_prompt = save_sample_negative_prompt
        self.n_save_sample = n_save_sample
        self.sample_seed = sample_seed
        self.save_guidance_scale = save_guidance_scale
        self.save_infer_steps = save_infer_steps

        self_dict = {
            "max_steps": self.max_steps,
            "instance_data_dir": self.instance_data_dir,
            "class_data_dir": self.class_data_dir,
            "file_prompt_contents": self.file_prompt_contents,
            "instance_prompt": self.instance_prompt,
            "class_prompt": self.class_prompt,
            "save_sample_prompt": self.save_sample_prompt,
            "instance_token": self.instance_token,
            "class_token": self.class_token,
            "num_class_image": self.num_class_image,
            "class_negative_prompt": self.class_negative_prompt,
            "class_guidance_scale": self.class_guidance_scale,
            "class_infer_steps": self.class_infer_steps,
            "save_sample_negative_prompt": self.save_sample_negative_prompt,
            "n_save_sample": self.n_save_sample,
            "sample_seed": self.sample_seed,
            "save_guidance_scale": self.save_guidance_scale,
            "save_infer_steps": self.save_infer_steps
        }
        super().__init__(self_dict)


