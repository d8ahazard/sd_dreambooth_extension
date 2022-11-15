import filecmp
import os
import shutil
import sys

import git

from launch import run
from modules.paths import script_path

dreambooth_skip_install = os.environ.get('DREAMBOOTH_SKIP_INSTALL', False)
if not dreambooth_skip_install:
    name = "Dreambooth"
    req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
    print(f"Installing requirements for Dreambooth")
    run(f'"{sys.executable}" -m pip install -r "{req_file}"', f"Checking {name} requirements.",
        f"Couldn't install {name} requirements.")

    # I think we only need to bump torch version to cu116 on Windows, as we're using prebuilt B&B Binaries...
    if os.name == "nt":
        torch_cmd = os.environ.get('TORCH_COMMAND', None)
        if torch_cmd is None:
            print("Checking/upgrading existing torch/torchvision installation")
            torch_cmd = "pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url " \
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

# Check for "different" B&B Files and copy only if necessary
if os.name == "nt":
    python = sys.executable
    bnb_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bitsandbytes_windows")
    bnb_dest = os.path.join(script_path, "venv", "Lib", "site-packages", "bitsandbytes")
    printed = False
    filecmp.clear_cache()
    for file in os.listdir(bnb_src):
        src_file = os.path.join(bnb_src, file)
        if file == "main.py":
            dest = os.path.join(bnb_dest, "cuda_setup")
        else:
            dest = bnb_dest
        dest_file = os.path.join(dest, file)
        if not os.path.exists(dest_file) or not filecmp.cmp(src_file, dest_file, True):
            if not printed:
                print("Copying 8Bit Adam files for Windows.")
                printed = True

            shutil.copy2(src_file, bnb_dest)
