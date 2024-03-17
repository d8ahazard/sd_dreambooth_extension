import os
from typing import Tuple, List, Dict

import gradio as gr

from dreambooth.utils.image_utils import FilenameTextGetter

image_data = []


def load_image_data(input_path: str, recurse: bool = False) -> List[Dict[str, str]]:
    if not os.path.exists(input_path):
        print(f"Input path {input_path} does not exist")
        return []
    global image_data
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
    image_data = output
    return output


def check_preprocess_path(input_path: str, recurse: bool = False) -> Tuple[gr.update, gr.update]:
    output_status = gr.update(visible=True)
    output_gallery = gr.update(visible=True)
    results = load_image_data(input_path, recurse)
    if len(results) == 0:
        return output_status, output_gallery
    else:
        images = [r['image'] for r in results]
        output_status = gr.update(visible=True, value='Found {len(results)} images')
        output_gallery = gr.update(visible=True, value=images)
        return output_status, output_gallery


def load_image_caption(evt: gr.SelectData):  # SelectData is a subclass of EventData
    return gr.update(value=f"You selected {evt.value} at {evt.index} from {evt.target}")
