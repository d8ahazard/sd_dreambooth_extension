import json
import os

from extensions.sd_dreambooth_extension.dreambooth import db_shared
from extensions.sd_dreambooth_extension.dreambooth.utils import get_images


class Concept(dict):
    def __init__(self, instance_data_dir: str = "", class_data_dir: str = "",
                 instance_prompt: str = "", class_prompt: str = "",
                 save_sample_prompt: str = "", save_sample_template: str = "", instance_token: str = "",
                 class_token: str = "", num_class_images: int = 0, num_class_images_per: int = 0,
                 class_negative_prompt: str = "", class_guidance_scale: float = 7.5, class_infer_steps: int = 60,
                 save_sample_negative_prompt: str = "", n_save_sample: int = 1, sample_seed: int = -1,
                 save_guidance_scale: float = 7.5, save_infer_steps: int = 60, input_dict=None):
        if input_dict is None:
            self.instance_data_dir = instance_data_dir
            self.class_data_dir = class_data_dir
            self.instance_prompt = instance_prompt
            self.class_prompt = class_prompt
            self.save_sample_prompt = save_sample_prompt
            self.save_sample_template = save_sample_template
            self.instance_token = instance_token
            self.class_token = class_token
            self.num_class_images = num_class_images
            self.num_class_images_per = num_class_images_per
            self.class_negative_prompt = class_negative_prompt
            self.class_guidance_scale = class_guidance_scale
            self.class_infer_steps = class_infer_steps
            self.save_sample_negative_prompt = save_sample_negative_prompt
            self.n_save_sample = n_save_sample
            self.sample_seed = sample_seed
            self.save_guidance_scale = save_guidance_scale
            self.save_infer_steps = save_infer_steps
        else:
            self.instance_data_dir = input_dict["instance_data_dir"] if "instance_data_dir" in input_dict else ""
            self.class_data_dir = input_dict["class_data_dir"] if "class_data_dir" in input_dict else ""
            self.instance_prompt = input_dict["instance_prompt"] if "instance_prompt" in input_dict else ""
            self.class_prompt = input_dict["class_prompt"] if "class_prompt" in input_dict else ""
            self.save_sample_prompt = input_dict["save_sample_prompt"] if "save_sample_prompt" in input_dict else ""
            self.save_sample_template = input_dict[
                "save_sample_template"] if "save_sample_template" in input_dict else ""
            self.instance_token = input_dict["instance_token"] if "instance_token" in input_dict else ""
            self.class_token = input_dict["class_token"] if "class_token" in input_dict else ""
            self.num_class_images = input_dict["num_class_images"] if "num_class_images" in input_dict else 0
            self.num_class_images_per = input_dict["num_class_images_per"] if "num_class_images_per" in input_dict else 0
            self.class_negative_prompt = input_dict[
                "class_negative_prompt"] if "class_negative_prompt" in input_dict else ""
            self.class_guidance_scale = input_dict[
                "class_guidance_scale"] if "class_guidance_scale" in input_dict else 7.5
            self.class_infer_steps = input_dict["class_infer_steps"] if "class_infer_steps" in input_dict else 60
            self.save_sample_negative_prompt = input_dict[
                "save_sample_negative_prompt"] if "save_sample_negative_prompt" in input_dict else ""
            self.n_save_sample = input_dict["n_save_sample"] if "n_save_sample" in input_dict else 1
            self.sample_seed = input_dict["sample_seed"] if "sample_seed" in input_dict else -1
            self.save_guidance_scale = input_dict["save_guidance_scale"] if "save_guidance_scale" in input_dict else 7.5
            self.save_infer_steps = input_dict["save_infer_steps"] if "save_infer_steps" in input_dict else 60

        if self.is_valid() and self.num_class_images != 0:
            if self.num_class_images_per == 0:
                images = get_images(self.instance_data_dir)
                if len(images) < self.num_class_images * 2:
                    self.num_class_images_per = 1
                else:
                    self.num_class_images_per = self.num_class_images // len(images)
                self.num_class_images = 0

        self_dict = {
            "instance_data_dir": self.instance_data_dir,
            "class_data_dir": self.class_data_dir,
            "instance_prompt": self.instance_prompt,
            "class_prompt": self.class_prompt,
            "save_sample_prompt": self.save_sample_prompt,
            "save_sample_template": self.save_sample_template,
            "instance_token": self.instance_token,
            "class_token": self.class_token,
            "num_class_images": self.num_class_images,
            "num_class_images_per": self.num_class_images_per,
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

def generate_concept_list(model_name: str,
                          instance_root: str,
                          class_root: str,
                          negative_prompt: str,
                          cfg_scale: float,
                          steps: int):
    # Check if the instance root directory exists
    if not os.path.exists(instance_root):
        print("The instance root directory does not exist: {}".format(instance_root))
        exit(1)

    # Initialize an empty list to hold the configurations
    configs = []

    # Iterate through the instance folders in the instance root directory
    for instance_folder in os.listdir(instance_root):
        # Get the full path to the instance folder
        instance_path = os.path.join(instance_root, instance_folder)
        # Check if the path is a directory (skip if not)
        if not os.path.isdir(instance_path):
            continue
        # Split the instance folder name into the instance name and the class name
        instance_name, class_name = instance_folder.split(' ')
        class_dir = os.path.join(class_root, class_name)
        if not os.path.exists(class_dir):
            class_dir = ""
        # Create a configuration for the instance and class
        concept = Concept(
            instance_path,
            class_dir,
            "[filewords]",
            "[filewords]",
            "[filewords]",
            "",
            instance_name,
            0,
            5,
            negative_prompt,
            cfg_scale,
            steps,
            negative_prompt
        )
        # Add the configuration to the list
        configs.append(concept.__dict__)
    # Output the configuration
    output = json.dumps(configs, ensure_ascii=False, indent=4)

    if db_shared.dreambooth_models_path is not None:
        out_dir = db_shared.dreambooth_models_path
    else:
        out_dir = os.path.join(db_shared.models_path, "dreambooth")
    out_file = os.path.join(out_dir, f"concepts_{model_name}")
    # Write the list of configurations to a JSON file
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(output)
    return output