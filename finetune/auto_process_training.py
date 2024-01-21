import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import traceback
from os.path import dirname

import torch
from typing import Dict

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if base_dir not in sys.path:
    print(f"Appending (install): {base_dir}")
    sys.path.insert(0, base_dir)
ext_dir = os.path.join(base_dir, 'extensions', 'sd_dreambooth_extension')
finetune_dir = os.path.join(ext_dir, 'finetune')
db_dir = os.path.join(finetune_dir, 'dreambooth')
# if ext_dir not in sys.path:
#     print(f"Appending (install): {ext_dir}")
#     sys.path.insert(0, ext_dir)
print(f"Appending (install): {finetune_dir}")
sys.path.insert(0, finetune_dir)
# if db_dir not in sys.path:
#     print(f"Appending (install): {db_dir}")
#     sys.path.insert(0, db_dir)

from modules.paths_internal import script_path

TRAINING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "finetune"))
TRAINING_DIR_TEMP = os.path.join(TRAINING_DIR, 'scripts')
TRAINING_CONFIGS_DIR = os.path.join(TRAINING_DIR, 'configs')
TRAINING_UI_DIR = os.path.join(TRAINING_DIR, 'ui')
TRAINING_CONVERSIONS_DIR = os.path.join(TRAINING_DIR, 'conversion')
TRAINING_GLOBAL_PARAMS_FILE = os.path.join(TRAINING_DIR, 'global_params.json')
HINT_JSON_FILE = os.path.join(script_path, "extensions", "sd_dreambooth_extension", "javascript", '_hints.js')

DIFFUSERS_REPO = "https://github.com/huggingface/diffusers.git"
TRAINING_SCRIPTS = [
    "examples/dreambooth/train_dreambooth.py",
    "examples/dreambooth/train_dreambooth_lora.py",
    "examples/dreambooth/train_dreambooth_lora_sdxl.py",
    "examples/text_to_image/train_text_to_image.py",
    "examples/text_to_image/train_text_to_image_lora.py",
    "examples/text_to_image/train_text_to_image_lora_sdxl.py",
    "examples/text_to_image/train_text_to_image_sdxl.py",
    "examples/controlnet/train_controlnet.py",
    "examples/controlnet/train_controlnet_sdxl.py",
    "examples/t2i_adapter/train_t2i_adapter_sdxl.py"
]

CONVERSION_SCRIPTS = [
    "scripts/convert_original_controlnet_to_diffusers.py",
    "scripts/convert_original_stable_diffusion_to_diffusers.py",
    "scripts/convert_diffusers_to_original_sdxl.py",
    "scripts/convert_diffusers_to_original_stable_diffusion.py",
    "scripts/convert_lora_safetensor_to_diffusers.py",
]

# If the OS is windows, fix the slashes in TRAINING_SCRIPTS
if os.name == 'nt':
    TRAINING_SCRIPTS = [file.replace('/', '\\') for file in TRAINING_SCRIPTS]


def generate_training_config_subclass(parse_args_func, output_file, converter=False):
    # Get the parser from the parse_args function
    parser = parse_args_func()
    assert isinstance(parser,
                      argparse.ArgumentParser), "parse_args_func must return an argparse.ArgumentParser instance."

    # Start building the class definition as a string

    file_name = os.path.splitext(os.path.basename(output_file))[0]
    class_name = file_name.replace('_', ' ').title().replace(' ', '')
    imports = "from pydantic import Field\nfrom finetune.dataclasses.base_config import BaseConfig\n\n"
    class_def = f"{imports}\nclass {class_name}(BaseConfig):\n"

    global_params = {}
    existing_params = {}
    if os.path.exists(TRAINING_GLOBAL_PARAMS_FILE):
        with open(TRAINING_GLOBAL_PARAMS_FILE, 'r') as f:
            existing_params = json.load(f)

    action_dest_keys = [action.dest for action in parser._actions]
    # Sort the keys alphabetically for consistent order
    action_dest_keys.sort()

    for key in action_dest_keys:
        # Retrieve the action by its destination (dest)
        action = next((a for a in parser._actions if a.dest == key), None)

        # Skip if the action is a help or version action
        if isinstance(action, (argparse._HelpAction, argparse._VersionAction)):
            continue
        if action.dest in existing_params:
            existing_action = existing_params[action.dest]
        else:
            existing_action = {}
        default = action.default if action.default is not argparse.SUPPRESS else None
        default = existing_action.get('default', default)
        arg_type = type(default).__name__ if default is not None else 'str'
        # Try parsing the default value as an int or float
        arg_type = existing_action.get('type', arg_type)
        # Format the field title
        title = action.dest.replace('_', ' ').title()
        title = existing_action.get('title', title)

        # Description from the help text
        description = action.help if action.help is not None else ''
        description = existing_action.get('description', description)

        # Add max_items for list types
        max_items = 1 if isinstance(action.type, list) else None

        field_extras = {
            'title': title,
            'description': description,
        }

        if max_items is not None:
            field_extras['max_items'] = max_items

        # Check for choices
        if action.choices is not None:
            field_extras['choices'] = action.choices

        existing_extras = existing_action.get('extras', {})
        for key, value in existing_extras.items():
            if key not in field_extras:
                field_extras[key] = value

        # Type constraints can be added similarly

        field_def = f"Field(default={repr(default)}"
        for k, v in field_extras.items():
            field_def += f", {k}={repr(v)}"
        field_def += ")"

        # Add the field to class definition
        class_def += f"    {action.dest}: {arg_type} = {field_def}\n"

        # Update global params
        global_params[action.dest] = {
            'default': default,
            'type': arg_type,
            'extras': field_extras,
        }

        existing_params[action.dest] = global_params[action.dest]

    # Write the global params to a file
    with open(TRAINING_GLOBAL_PARAMS_FILE, 'w', encoding='utf-8') as file:
        json.dump(existing_params, file, indent=4)

    # Write the class definition to a file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(class_def)

    create_ui_files(global_params, file_name)


def load_parse_args_from_file(file_path):
    try:
        spec = importlib.util.spec_from_file_location("module.name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        print(f"ImportError Traceback:")
        traceback.print_exc()
        return None

    return module.parse_args if hasattr(module, 'parse_args') else None


def disable_min_version_check(file):
    # Find the below line in the file and comment it out if it's not already commented out
    # check_min_version("0.24.0.dev0")
    print(f"Disabling min version check in {file}")
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(file, 'w') as f:
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('check_min_version('):
                print("Found min version check, removing.")
                continue
            f.write(line)


def disable_main_launch_lines(file):
    # Find the below line in the file and comment it out if it's not already commented out
    # if __name__ == '__main__':
    #     main()
    print(f"Disabling main launch lines in {file}")
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(file, 'w') as f:
        do_comment = False
        for line in lines:
            # if __name__ == "__main__":
            stripped_line = line.strip()
            if stripped_line.startswith('if __name__ == \'__main__\':') or stripped_line.startswith(
                    'if __name__ == "__main__":'):
                print("Found main launch line")
                break
            f.write(line)


def replace_main_conversion_lines_init(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    in_args = False
    import_added = False
    config_class_name = f"{os.path.basename(file).replace('.py', '')}_config".replace('_', ' ').title().replace(' ',
                                                                                                                '')
    config_module_name = f"{os.path.basename(file).replace('.py', '')}_config"
    with open(file, 'w') as f:
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('if __name__ == \'__main__\':') or stripped_line.startswith(
                    'if __name__ == "__main__":'):
                print("Found main launch line")
                line = "def parse_args():\n"
                in_args = True
            elif in_args and stripped_line.startswith("args = parser.parse_args()"):
                line = f"    return parser\n\n\ndef convert():"
                in_args = False
            f.write(line)


def replace_main_conversion_lines(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    import_added = False
    config_class_name = f"{os.path.basename(file).replace('.py', '')}_config".replace('_', ' ').title().replace(' ',
                                                                                                                '')
    config_module_name = f"{os.path.basename(file).replace('.py', '')}_config"
    import_line = f"from finetune.configs.{config_module_name} import {config_class_name}\n"
    with open(file, 'w') as f:
        for line in lines:
            stripped_line = line.strip()
            if stripped_line == import_line:
                import_added = True
            if stripped_line.startswith('import') and not import_added:
                line = import_line + line
                import_added = True
            if stripped_line.startswith('def convert()'):
                line = line.replace('def convert()', f"def convert(args: {config_class_name})")
            f.write(line)


def disable_parse_args_lines(file):
    print(f"Modifying {file}")
    with open(file, 'r') as f:
        lines = f.readlines()

    in_parse_args = False
    found_add_argument = False
    parse_method_end_line = -1
    add_args_end_line = -1
    line_number = 0

    for line in lines:
        stripped_line = line.strip()

        # Detect the start of the parse_args function
        if 'def parse_args' in stripped_line:
            in_parse_args = True

        # Process lines within parse_args function
        if in_parse_args:
            if "def " in stripped_line and not "def parse_args" in stripped_line:
                # Found a new function, parse_args must be over
                parse_method_end_line = line_number - 1
                in_parse_args = False
            if "return" in stripped_line:
                # Found the return statement, parse_args must be over
                parse_method_end_line = line_number
                in_parse_args = False
            elif "add_argument" in stripped_line:
                found_add_argument = True
            if found_add_argument and ")" in stripped_line and not ")," in stripped_line:
                add_args_end_line = line_number
                found_add_argument = False
        line_number += 1
    line_number = 0
    with open(file, 'w') as f:
        for line in lines:
            if "return args" in line:
                line = line.replace("return args", "return parser")

            if add_args_end_line < line_number < parse_method_end_line:
                line = ""
            f.write(line)
            line_number += 1


def add_global_args(file):
    # Find the line logger = logging.getLogger(__name__) and add the below lines after it
    # args = None
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(file, 'w') as f:
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('logger = logging.getLogger(__name__)') or stripped_line.startswith(
                    "logger = get_logger(__name__)"):
                line = f"{line}\nargs = None\nglobal_step = 0\nstatus = FinetuneStatus()\n"
            f.write(line)


def fix_main_method(file):
    # Replace "def main()" with "def main(args)",
    # and replace args = parse_args() with #args = parse_args()
    print(f"Fixing main method in {file}")
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(file, 'w') as f:
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('def main()') or stripped_line.startswith('def main(args)'):
                line = line.replace('def main()', 'def main(user_args)')
                line = line.replace('def main(args)', 'def main(user_args)')
                indent = len(line) - len(line.lstrip())
                line += f"\n{' ' * 4}global global_step\n"
                line += f"{' ' * 4}global args\n"
                line += f"{' ' * 4}args = user_args\n"
            elif stripped_line.startswith('args = parse_args()'):
                # Get the indent level
                continue
            f.write(line)


def inject_cancel_check(file):
    # if global_step >= args.max_train_steps:
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(file, 'w') as f:
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('if global_step >= args.max_train_steps:'):
                print(f"Injecting cancel check into {file}")
                line = line.replace('if global_step >= args.max_train_steps:',
                                    'if global_step >= args.max_train_steps or status.state == FinetuneState.CANCELLED:')
                indent = len(line) - len(line.lstrip())
                line_addition = (f"\n{' ' * indent}{' ' * 4}print(f\"Cancelled at step {{global_step}}\")\n"
                                 f"{' ' * indent}{' ' * 4}status.status = f\"Cancelled at step {{global_step}}\"\n")
                line = line + line_addition
            f.write(line)


def swap_imports(file):
    # Replace from tqdm.auto import tqdm with from helpers.mytqdm import mytqdm as tqdm
    print(f"Swapping imports in {file}")
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(file, 'w') as f:
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('from tqdm.auto import tqdm'):
                line = line.replace('from tqdm.auto import tqdm', 'from finetune.helpers.ft_tqdm import fttqdm as tqdm')
                indent = len(line) - len(line.lstrip())
                line += f"\n{' ' * indent}from finetune.helpers.ft_state import FinetuneStatus, FinetuneState\n"
                line += f"{' ' * indent}from dreambooth.utils.image_utils import db_save_image\n"
            f.write(line)


def inject_log_validation(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(file, 'w') as f:
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('images.append(image)'):
                print(f"Injecting log_validation into {file}")
                indent = len(line) - len(line.lstrip()) - 4
                line_addition = (f"\n{' ' * indent}image_paths = []\n"
                                 f"{' ' * indent}index = 0\n"
                                 f"{' ' * indent}status.clear_samples()\n"
                                 f"{' ' * indent}for image in images:\n"
                                 f"{' ' * indent}{' ' * 4}image_path = db_save_image(\n"
                                 f"{' ' * indent}{' ' * 4}{' ' * 4}image,\n"
                                 f"{' ' * indent}{' ' * 4}{' ' * 4}args.validation_prompt,\n"
                                 f"{' ' * indent}{' ' * 4}{' ' * 4}custom_name=f\"sample_{{global_step}}-{{index}}\",\n"
                                 f"{' ' * indent}{' ' * 4})\n"
                                 f"{' ' * indent}{' ' * 4}image_paths.append(image_path)\n"
                                 f"{' ' * indent}{' ' * 4}index += 1\n"
                                 f"{' ' * indent}{' ' * 4}status.add_sample(image_path, args.validation_prompt)\n")
                line += line_addition
            f.write(line)


def create_gradio_textfield(field: dict):
    element_label = field['extras']['title']
    element_default = field['default']
    element_visible = field['extras'].get('visible', True)

    return f"gr.Textbox(interactive=True, label='{element_label}', value='{element_default}', visible={element_visible})"


def create_gradio_slider(field: dict):
    element_label = field['extras']['title']
    element_default = field['default']
    element_visible = field['extras'].get('visible', True)
    default_max = 100 if field['type'] == 'int' else 1
    element_min = field['extras'].get('min', 0)
    element_max = field['extras'].get('max', default_max)
    default_step = 1 if field['type'] == 'int' else 0.01
    element_step = field['extras'].get('step', default_step)
    return f"gr.Slider(interactive=True, label='{element_label}', value={element_default}, visible={element_visible}, step={element_step}, minimum={element_min}, maximum={element_max})"


def create_gradio_checkbox(field: dict):
    element_label = field['extras']['title']
    element_default = field['default'] if isinstance(field['default'], bool) else False
    element_visible = field['extras'].get('visible', True)
    return f"gr.Checkbox(interactive=True, label='{element_label}', value={element_default}, visible={element_visible})"


def create_gradio_select(field: dict):
    element_label = field['extras']['title']
    element_default = field['default']
    element_visible = field['extras'].get('visible', True)
    element_choices = field['extras']['choices']
    return f"gr.Dropdown(interactive=True, label='{element_label}', choices={element_choices}, value='{element_default}', visible={element_visible})"


def create_gradio_element(field_name, elem_type, field: Dict):
    # Escape quotes in the field description
    field['extras']['description'] = field['extras']['description'].replace("'", "\\'")
    if elem_type == "str":
        if field["extras"].get("choices"):
            return f"{field_name.upper()} = " + create_gradio_select(field)
        else:
            return f"{field_name.upper()} = " + create_gradio_textfield(field)
    elif elem_type == "int" or elem_type == "float":
        return f"{field_name.upper()} = " + create_gradio_slider(field)
    elif elem_type == "bool":
        return f"{field_name.upper()} = " + create_gradio_checkbox(field)
    else:
        print(f"Unknown field type: {elem_type}")
        return None


def create_ui_files(params_dict, file):
    file = file.replace('config', 'ui')
    ui_file_name = file + '.py'
    ui_file_path = os.path.join(TRAINING_UI_DIR, ui_file_name)
    print(f"Creating UI files for {file} at {ui_file_path}")

    grouped_elements = {}
    for field_name, data in params_dict.items():
        ignore = data["extras"].get('ignore', False)
        if ignore:
            continue
        group_name = data["extras"].get('group', 'Other')
        if group_name not in grouped_elements:
            grouped_elements[group_name] = {}
        grouped_elements[group_name][field_name] = data
    global_declaration_lines = []
    global_import_lines = []
    register_element_lines = []
    header_lines = ["import gradio as gr", "", "from dreambooth.utils.ui_utils import ElementManager", "",
                    "manager = ElementManager()", ""]
    render_lines = []
    sorted_group_names = sorted(grouped_elements.keys())
    for group_name in sorted_group_names:
        group = grouped_elements[group_name]
        indent = 4
        group_name = group_name.replace(' ', '_').upper()
        group_accordion_line = f"{' ' * indent}with gr.Accordion(open=False, label=\"{group_name.replace('_', ' ').title()}\") as {group_name}_ACCORDION:"
        render_lines.append(group_accordion_line)
        indent = 8
        for field_name, data in group.items():
            description = data["extras"].get('description', '')
            advanced = data["extras"].get('advanced', False)
            field_type = data['type']
            element = create_gradio_element(field_name, field_type, data)
            if element:
                global_declaration_lines.append(f"{field_name.upper()} = None")
                global_import_lines.append(f"{' ' * 4}global {field_name.upper()}")
                reg_string = file.replace("_ui", "")
                description_escaped = description.replace("'", "\\'")
                description_escaped = description_escaped.replace('"', '\\"')
                register_element_lines.append(
                    f"{' ' * 4}manager.register_db_component(\"{reg_string}\", {field_name.upper()}, \"{field_name}\", {advanced}, \"{description_escaped}\")")
                render_lines.append(f"{' ' * indent}{element}")

    ui_lines = header_lines + global_declaration_lines + ["", "",
                                                          "def render():"] + global_import_lines + render_lines + register_element_lines

    with open(ui_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ui_lines))


def process_scripts():
    # Clone the diffusers repo to a temp path
    temp_path = tempfile.mkdtemp()
    os.makedirs(TRAINING_DIR_TEMP, exist_ok=True)
    os.makedirs(TRAINING_CONFIGS_DIR, exist_ok=True)
    os.makedirs(TRAINING_CONFIGS_DIR, exist_ok=True)
    os.makedirs(TRAINING_UI_DIR, exist_ok=True)
    os.system(f"git clone {DIFFUSERS_REPO} {temp_path}")
    for file in TRAINING_SCRIPTS:
        output_file_name = os.path.splitext(os.path.basename(file))[0] + '_config.py'
        output_file_path = os.path.join(TRAINING_CONFIGS_DIR, output_file_name)
        file_path = os.path.join(temp_path, file)
        if os.path.exists(file_path):
            print(f"Processing {file_path}")
            # Copy the file to TRAINING_DIR_TEMP
            dest_file_path = os.path.join(TRAINING_DIR_TEMP, os.path.basename(file_path))
            if os.path.exists(dest_file_path):
                os.remove(dest_file_path)
            shutil.copy(file_path, TRAINING_DIR_TEMP)
            file_path = dest_file_path
            try:
                # Disable min version check
                disable_min_version_check(file_path)
                disable_main_launch_lines(file_path)
                disable_parse_args_lines(file_path)
                try:
                    print(f"Loading parse_args function from {file_path}")
                    parse_args_func = load_parse_args_from_file(file_path)
                    print(f"Loaded parse_args function: {parse_args_func}")
                    if parse_args_func:
                        print(f"Generating config for {file} at {output_file_path}")
                        generate_training_config_subclass(parse_args_func, output_file_path)
                        print(f"Generated config for {file} at {output_file_path}")
                    else:
                        print(f"No parse_args function found in {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    traceback.print_exc()
                add_global_args(file_path)
                fix_main_method(file_path)
                swap_imports(file_path)
                inject_log_validation(file_path)
                inject_cancel_check(file_path)
            except:
                print(f"Error disabling min version check in {file}")
                traceback.print_exc()



        else:
            print(f"File not found: {file_path}")
    for file in CONVERSION_SCRIPTS:
        file_path = os.path.join(temp_path, file)
        dest_path = os.path.join(TRAINING_CONVERSIONS_DIR, os.path.basename(file_path))
        if os.path.exists(file_path):
            print(f"Copying {file}")
            # Remove the file if it already exists
            if os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.copy(file_path, dest_path)
            file_path = dest_path
            replace_main_conversion_lines_init(dest_path)

            try:
                print(f"Loading parse_args function from {file_path}")
                parse_args_func = load_parse_args_from_file(file_path)
                print(f"Loaded parse_args function: {parse_args_func}")
                if parse_args_func:
                    out_script_name = f"{file_path.replace('.py', '')}_config.py"
                    output_file_path = os.path.join(TRAINING_CONFIGS_DIR, os.path.basename(out_script_name))
                    print(f"Generating (convert)config for {file} at {output_file_path}")
                    generate_training_config_subclass(parse_args_func, output_file_path)
                    print(f"Generated config for {file} at {output_file_path}")
                else:
                    print(f"No parse_args function found in {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
                traceback.print_exc()
            replace_main_conversion_lines(dest_path)

    try:
        os.rmdir(temp_path)
    except:
        pass


if __name__ == '__main__':
    process_scripts()
