import json
import os
from pydantic import BaseModel

from dreambooth import shared


class BaseConfig(BaseModel):

    @classmethod
    def from_dict(cls, data):
        # Use cls(**data) to create an instance of the class from a dictionary
        return cls(**data)

    def to_dict(self):
        # Convert the model instance into a dictionary
        return self.diact()

    def to_json(self):
        # Serialize the dictionary form of the instance to a JSON string
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        # Create an instance of the class from a JSON string
        return cls.parse_raw(json_str)

    @classmethod
    def from_file(cls, file_path):
        # Load a JSON file and create an instance of the class from it
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, workspace_name):
        # Save the instance of this class to a JSON file
        file_path = os.path.join(shared.models_path, 'finetune', workspace_name)

        # Get the name of this class
        class_name = self.__class__.__name__
        file_path = os.path.join(file_path, class_name + '.json')

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write the JSON representation of the instance to the file
        with open(file_path, 'w') as f:
            f.write(self.to_json())

