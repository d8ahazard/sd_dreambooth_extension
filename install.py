import shutil
import subprocess
import sys
import importlib
import git

from launch import run_pip, run
import os
from modules.paths import script_path
name = "Dreambooth"
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
print(f"loading Dreambooth reqs from {req_file}")
run(f'"{sys.executable}" -m pip install -r "{req_file}"', "Checking Dreambooth requirements.", "Couldn't install Dreambooth requirements.")
torch_cmd="pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url " \
          "https://download.pytorch.org/whl/cu116 "
run(f'"{sys.executable}" -m {torch_cmd}', "Checking torch and torchvision versions", "Couldn't install torch")

try:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    repo = git.Repo(base_dir)
    revision = repo.rev_parse("HEAD")
    print(f"Dreambooth revision is {revision}")

    import diffusers
    import torch
    import torchvision
    ver = diffusers.__version__
    tver = torch.__version__
    tvver = torchvision.__version__
    print(f"Diffusers version is {ver}.")
    print(f"Torch version is {tver}.")
    print(f"Torch vision version is {tvver}.")
except Exception as e:
    print(f"Exception doing the things: {e}")
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
