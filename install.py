import shutil
import sys
import importlib
import git

from launch import run_pip, run
import os
from modules.paths import script_path
base_dir = os.path.dirname(os.path.realpath(__file__))
try:
    repo = git.Repo(base_dir)
    revision = repo.rev_parse("HEAD")
    print(f"Dreambooth revision is {revision}")
except:
    pass

name = "Dreambooth"
reqs = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
python = sys.executable
run(f'{python} -m pip install -r "{reqs}"', "Checking Dreambooth requirements",
    "Installing Dreambooth requirements failed.")
print(f"Dreambooth req install: {reqs}")
torch_cmd="pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url " \
          "https://download.pytorch.org/whl/cu116 "
run(f'"{python}" -m {torch_cmd}', "Checking Dreambooth torch version", "Updating torch version failed!")

try:
    import diffusers
    import torch
    import torchvision
    importlib.reload(diffusers)
    importlib.reload(torch)
    importlib.reload(torchvision)
    ver = diffusers.__version__
    tver = torch.__version__
    tvver = torchvision.__version__
    print(f"Diffusers version is {ver}.")
    print(f"Torch version is {tver}.")
    print(f"Torch vision version is {tvver}.")
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
