import shutil
import sys

from launch import run_pip, run
import os
from modules.paths import script_path

name = "Dreambooth"
reqs = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
python = sys.executable
run(f'{python} -m pip install -r "{reqs}"', desc=f"Installing Requirements for {name}", errdesc=f"Couldn't install {name} requirements.")

try:
    import diffusers

    ver = diffusers.__version__
    print(f"Diffusers version is {ver}.")
except:
    pass
if os.name == "nt":
    python = sys.executable
    bnb_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bitsandbytes_windows")
    site_dir = os.path.join(script_path, "venv", "Lib", "site-packages", "bitsandbytes")
    print("Copying 8Bit Adam files for Windows.")
    for file in os.listdir(bnb_dir):
        fullfile = os.path.join(bnb_dir, file)
        if file == "main.py":
            shutil.copy(fullfile, os.path.join(site_dir, "cuda_setup"))
        else:
            shutil.copy(fullfile, site_dir)
