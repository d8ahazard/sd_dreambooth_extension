import os


class Concept(dict):
    def __init__(self, max_steps: int = -1, instance_data_dir: str = "", class_data_dir: str = "",
                 instance_prompt: str = "", class_prompt: str = "",
                 save_sample_prompt: str = "", save_sample_template: str = "", instance_token: str = "",
                 class_token: str = "", num_class_images: int = 0, class_negative_prompt: str = "",
                 class_guidance_scale: float = 7.5, class_infer_steps: int = 60, save_sample_negative_prompt: str = "",
                 n_save_sample: int = 1, sample_seed: int = -1, save_guidance_scale: float = 7.5,
                 save_infer_steps: int = 60, input_dict=None):
        if input_dict is None:
            self.max_steps = max_steps
            self.instance_data_dir = instance_data_dir
            self.class_data_dir = class_data_dir
            self.instance_prompt = instance_prompt
            self.class_prompt = class_prompt
            self.save_sample_prompt = save_sample_prompt
            self.save_sample_template = save_sample_template
            self.instance_token = instance_token
            self.class_token = class_token
            self.num_class_images = num_class_images
            self.class_negative_prompt = class_negative_prompt
            self.class_guidance_scale = class_guidance_scale
            self.class_infer_steps = class_infer_steps
            self.save_sample_negative_prompt = save_sample_negative_prompt
            self.n_save_sample = n_save_sample
            self.sample_seed = sample_seed
            self.save_guidance_scale = save_guidance_scale
            self.save_infer_steps = save_infer_steps
        else:
            self.max_steps = input_dict["max_steps"] if "max_steps" in input_dict else -1
            self.instance_data_dir = input_dict["instance_data_dir"] if "instance_data_dir" in input_dict else ""
            self.class_data_dir = input_dict["class_data_dir"] if "class_data_dir" in input_dict else ""
            self.instance_prompt = input_dict["instance_prompt"] if "instance_prompt" in input_dict else ""
            self.class_prompt = input_dict["class_prompt"] if "class_prompt" in input_dict else ""
            self.save_sample_prompt = input_dict["save_sample_prompt"] if "save_sample_prompt" in input_dict else ""
            self.save_sample_template = input_dict["save_sample_template"] if "save_sample_template" in input_dict else ""
            self.instance_token = input_dict["instance_token"] if "instance_token" in input_dict else ""
            self.class_token = input_dict["class_token"] if "class_token" in input_dict else ""
            self.num_class_images = input_dict["num_class_images"] if "num_class_images" in input_dict else 0
            self.class_negative_prompt = input_dict["class_negative_prompt"] if "class_negative_prompt" in input_dict else ""
            self.class_guidance_scale = input_dict["class_guidance_scale"] if "class_guidance_scale" in input_dict else 7.5
            self.class_infer_steps = input_dict["class_infer_steps"] if "class_infer_steps" in input_dict else 60
            self.save_sample_negative_prompt = input_dict["save_sample_negative_prompt"] if "save_sample_negative_prompt" in input_dict else ""
            self.n_save_sample = input_dict["n_save_sample"] if "n_save_sample" in input_dict else 1
            self.sample_seed = input_dict["sample_seed"] if "sample_seed" in input_dict else -1
            self.save_guidance_scale = input_dict["save_guidance_scale"] if "save_guidance_scale" in input_dict else 7.5
            self.save_infer_steps = input_dict["save_infer_steps"] if "save_infer_steps" in input_dict else 60

        self_dict = {
            "max_steps": self.max_steps,
            "instance_data_dir": self.instance_data_dir,
            "class_data_dir": self.class_data_dir,
            "instance_prompt": self.instance_prompt,
            "class_prompt": self.class_prompt,
            "save_sample_prompt": self.save_sample_prompt,
            "save_sample_template": self.save_sample_template,
            "instance_token": self.instance_token,
            "class_token": self.class_token,
            "num_class_images": self.num_class_images,
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

    def is_valid(self):
        if self.instance_data_dir is not None and self.instance_data_dir != "":
            if os.path.exists(self.instance_data_dir):
                return True
        return False
