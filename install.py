import filecmp
import importlib.util
import os
import shutil
import sys
import sysconfig

import git
from launch import run

from modules.paths import script_path

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")


def check_versions():
    global req_file
    reqs = open(req_file, 'r')
    lines = reqs.readlines()
    reqs_dict = {}
    for line in lines:
        splits = line.split("==")
        if len(splits) == 2:
            key = splits[0]
            if "torch" not in key:
                if "diffusers" in key:
                    key = "diffusers"
                reqs_dict[key] = splits[1].replace("\n", "").strip()
    if os.name == "nt":
        reqs_dict["torch"] = "1.12.1+cu116"
        reqs_dict["torchvision"] = "0.13.1+cu116"

    checks = ["bitsandbytes", "diffusers", "transformers", "torch", "torchvision", "xformers"]
    for check in checks:
        check_ver = "N/A"
        status = "[ ]"
        try:
            check_available = importlib.util.find_spec(check) is not None
            if check_available:
                check_ver = importlib_metadata.version(check)
                if check in reqs_dict:
                    req_version = reqs_dict[check]
                    if str(check_ver) == str(req_version):
                        status = "[+]"
                    else:
                        status = "[!]"
        except importlib_metadata.PackageNotFoundError:
            check_available = False
        if not check_available:
            status = "[!]"
            print(f"{status} {check} NOT installed.")
        else:
            print(f"{status} {check} version {check_ver} installed.")


dreambooth_skip_install = os.environ.get('DREAMBOOTH_SKIP_INSTALL', False)

if not dreambooth_skip_install:
    name = "Dreambooth"    
    run(f'"{sys.executable}" -m pip install -r "{req_file}"', f"Checking {name} requirements...",
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
check_versions()
# Check for "different" B&B Files and copy only if necessary
if os.name == "nt":
    python = sys.executable
    bnb_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bitsandbytes_windows")
    bnb_dest = os.path.join(sysconfig.get_paths()["purelib"], "bitsandbytes")
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
