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


base_dir = os.path.dirname(os.path.realpath(__file__))
repo = git.Repo(base_dir)
revision = repo.rev_parse("HEAD")
print(f"Dreambooth revision is {revision}")

try:
    import diffusers
    import torch
    import torchvision
    import transformers
    from diffusers.utils.import_utils import is_xformers_available
    diffusers_ver = diffusers.__version__
    diffusers_rec = "0.7.2"
    torch_ver = torch.__version__
    torch_rec = "1.12.1+cu116"
    torchvision_ver = torchvision.__version__
    torchvision_rec = "0.13.1+cu116"
    transformers_ver = transformers.__version__
    transformers_rec = "4.21.0"
    torch_flag = False
    vis_flag = False
    if os.name == "nt":
        if torch_rec != torch_ver:
            torch_flag = True
        if torchvision_ver != torchvision_rec:
            vis_flag = True

    has_xformers = False
    try:
        args = sys.argv
        print(f"Args: {args}")
        if is_xformers_available():
            import xformers
            import xformers.ops
            has_xformers = True
    except:
        pass

    print(f"[{'*' if diffusers_rec == diffusers_ver else '!'}] Diffusers version is {diffusers_ver}.")
    print(f"[{'*' if not torch_flag else '!'}] Torch version is {torch_ver}.")
    print(f"[{'*' if not vis_flag else '!'}] Torch vision version is {torchvision_ver}.")
    print(f"[{'*' if transformers_ver == transformers_rec else '!'}] Transformers version is {transformers_ver}.")
    print(f"[{'*' if has_xformers else '-'}] Xformers")
    print(f"")
except:
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
        shutil.copy2(src_file, dest)
