import json
import logging
import os

from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    model_dir: str = Field("sd-model", description="Base path of the model.")
    model_name: str = Field("sd-model", description="Name of the model.")
    revision: int = Field(0, description="Revision number of the model.")
    config_prefix: str = Field("", description="Prefix for the config file.")
    pretrained_model_name_or_path: str = Field("", description="Path to the pretrained model.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "models_path" in kwargs and "model_name" in kwargs:
            model_path = os.path.join(kwargs["models_path"], kwargs["model_name"])
            self.model_dir = model_path
        for k, v in kwargs.items():
            if hasattr(self, k) and isinstance(v, type(getattr(self, k))):
                setattr(self, k, v)
        if self.model_dir != "" and self.pretrained_model_name_or_path == "":
            self.pretrained_model_name_or_path = os.path.join(self.model_dir, "working")

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
        config_file = os.path.join(models_path, f"{self.config_prefix}_config.json")

        if backup:
            backup_dir = os.path.join(models_path, "backups")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            config_file = os.path.join(models_path, "backups", f"db_config_{self.revision}.json")

        with open(config_file, "w") as outfile:
            json.dump(self.__dict__, outfile, indent=4)

    def get_params(self):
        tc_fields = {}
        for f, data in self.__fields__.items():
            value = getattr(self, f)
            try:
                json.dumps(value)
            except TypeError:
                continue
            field_dict = {}
            for prop in ['default', 'description', 'choices', 'title']:
                if hasattr(data.field_info, prop):
                    value = getattr(data.field_info, prop)
                    # Check if the property is JSON serializable
                    try:
                        json.dumps(value)
                        field_dict[prop] = value
                    except TypeError:
                        pass
            field_dict['value'] = getattr(self, f)
            field_dict['type'] = data.outer_type_.__name__
            tc_fields[f] = field_dict
        return tc_fields

    def load_params(self, params_dict):
        for key, value in params_dict.items():
            try:
                if f"{self.config_prefix}_" in key:
                    key = key.replace(f"{self.config_prefix}_", "")
                if hasattr(self, key):
                    setattr(self, key, value)
            except Exception as e:
                print(e)

    def load_from_file(self, model_dir: str):
        """
        Load the config from a JSON file
        """
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            logging.getLogger(__name__).info("Model directory doesn't exist, can't create or load config.")
            return
        # check if the file exists

        model_file = os.path.join(model_dir, f"{self.config_prefix}_config.json")
        if not os.path.exists(model_file):
            logging.getLogger(__name__).info("Config doesn't exist.")
            self.save()
        else:
            # load the JSON file
            with open(model_file, "r") as infile:
                data = json.load(infile)

            # update the instance variables
            self.load_params(data)
        if self.pretrained_model_name_or_path == "":
            self.pretrained_model_name_or_path = os.path.join(model_dir, "working")
        return self
