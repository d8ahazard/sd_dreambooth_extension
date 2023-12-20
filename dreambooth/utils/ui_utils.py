import json
import os
from typing import Dict, Any

from dreambooth.shared import script_path


class ElementManager:
    elements = {}
    advanced_elements = {}
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print('Creating the object')
            cls._instance = super(ElementManager, cls).__new__(cls)
            # Put any initialization here.
        return cls._instance

    def register_db_component(self, trainer, element, name, advanced=False, description=None):
        if trainer not in self.elements:
            self.elements[trainer] = {}
        setattr(element, "do_not_save_to_config", True)
        self.elements[trainer][name] = element
        if advanced:
            if trainer not in self.advanced_elements:
                self.advanced_elements[trainer] = {}
            self.advanced_elements[trainer][name] = element
        hint_file = os.path.join(script_path, "extensions", "sd_dreambooth_extension", "javascript", '_hints.js')
        # Read the hints.js file
        hint_string = "let more_hints = "
        if os.path.exists(hint_file):
            with open(hint_file, 'r') as f:
                content = f.read()
            # Remove the hing_string from content
            content = content.replace(hint_string, "")
            # Parse the json
            hints = json.loads(content.strip())
        else:
            hints = {}
        # Get the label or value of the element
        if hasattr(element, "label"):
            label = element.label
        elif hasattr(element, "value"):
            label = element.value
        else:
            label = name
        # Add the hint
        hints[label] = description
        # Write the hints back to the file
        with open(hint_file, 'w') as f:
            f.write(hint_string + json.dumps(hints, indent=4))

    def get_advanced_elements(self):
        advanced_elements_list = []
        for trainer in self.advanced_elements:
            for name in self.advanced_elements[trainer]:
                advanced_elements_list.append(self.advanced_elements[trainer][name])
        return advanced_elements_list

    def get_elements(self, trainer) -> Dict[str, Any]:
        elements_list = self.elements.get(trainer, {})
        return elements_list
