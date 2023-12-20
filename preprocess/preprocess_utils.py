import inspect
import os
import pkgutil
from typing import Tuple, List, Dict

import gradio as gr

import preprocess
from dreambooth.utils.image_utils import FilenameTextGetter

all_image_data = []
selected_image_data = []
current_index = -1

def load_all_image_data(input_path: str, recurse: bool = False) -> List[Dict[str,str]]:
    if not os.path.exists(input_path):
        print(f"Input path {input_path} does not exist")
        return []
    global all_image_data
    results = []
    from dreambooth.utils.image_utils import list_features, is_image
    pil_features = list_features()
    # Get a list from PIL of all the image formats it supports

    for root, dirs, files in os.walk(input_path):
        for file in files:
            full_path = os.path.join(root, file)
            print(f"Checking {full_path}")
            if is_image(full_path, pil_features):
                results.append(full_path)
        if not recurse:
            break

    output = []
    text_getter = FilenameTextGetter()
    for img_path in results:
        file_text = text_getter.read_text(img_path)
        output.append({'image': img_path, 'text': file_text})
    all_image_data = output
    return output

def check_preprocess_path(input_path: str, recurse: bool = False) -> Tuple[gr.update, gr.update]:
    output_status = gr.update(visible=True)
    output_gallery = gr.update(visible=True)
    results = load_all_image_data(input_path, recurse)
    if len(results) == 0:
        return output_status, output_gallery
    else:
        images = [r['image'] for r in results]
        output_status = gr.update(visible=True, value=f"Found {len(results)} images")
        output_gallery = gr.update(visible=True, value=images)
        return output_status, output_gallery

def load_image_caption(evt: gr.SelectData):  # SelectData is a subclass of EventData
    global current_index
    current_index = evt.index
    if len(all_image_data) <= evt.index:
        return gr.update(value=f"Index {evt.index} is out of range")
    else:
        current_index = evt.index
        return gr.update(value=all_image_data[evt.index]['text'])


def get_processors() -> List[str]:
    # Find all classes that inherit from BaseCaptioner
    from preprocess.captioners.base import BaseCaptioner
    processors = []

    # Function to recursively find subclasses in a package
    def find_subclasses(module):
        for loader, mod_name, is_pkg in pkgutil.walk_packages(module.__path__):
            # Load the module
            mod = loader.find_module(mod_name).load_module(mod_name)
            for name, obj in inspect.getmembers(mod):
                if inspect.isclass(obj) and issubclass(obj, BaseCaptioner) and obj is not BaseCaptioner:
                    processors.append(name)

            # If it's a package, recurse
            if is_pkg:
                find_subclasses(mod)

    # Start searching from the base captioners package
    find_subclasses(preprocess.captioners)

    return processors

def get_processor(name: str):
    # Find all classes that inherit from BaseCaptioner
    from preprocess.captioners.base import BaseCaptioner
    processors = []

    # Function to recursively find subclasses in a package
    def find_subclasses(module):
        for loader, mod_name, is_pkg in pkgutil.walk_packages(module.__path__):
            # Load the module
            mod = loader.find_module(mod_name).load_module(mod_name)
            for obj_name, obj in inspect.getmembers(mod):
                if inspect.isclass(obj) and issubclass(obj, BaseCaptioner) and obj is not BaseCaptioner:
                    if obj_name == name:
                        return obj

            # If it's a package, recurse
            if is_pkg:
                find_subclasses(mod)

    # Start searching from the base captioners package
    return find_subclasses(preprocess.captioners)

def caption_images():
    pass