import filecmp
import importlib.util
import os
import shutil
import sys
import sysconfig

import git

from launch import run

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
torch_cmd = "pip install --no-deps https://download.pytorch.org/whl/nightly/cu118/torch-2.0.0.dev20230202%2Bcu118-cp310-cp310-win_amd64.whl https://download.pytorch.org/whl/nightly/cu118/torchvision-0.15.0.dev20230202%2Bcu118-cp310-cp310-win_amd64.whl"

xformers_cmd = "pip install https://github.com/ArrowM/xformers/releases/download/xformers-0.0.17-cu118-feb-02-23/xformers-0.0.17+48a77cc.d20230202-cp310-cp310-win_amd64.whl"

def fix_torch():
    try:
        run(f'"{python}" -m {torch_cmd}', "Installing torch2 and torchvision.", "Couldn't install torch.")
        has_torch = importlib.util.find_spec("torch") is not None
        has_torch_vision = importlib.util.find_spec("torchvision") is not None

        torch_check = str(importlib_metadata.version("torch")) if has_torch else None
        torch_vision_check = str(importlib_metadata.version("torchvision")) if has_torch_vision else None
        return torch_check, torch_vision_check
    except Exception as e:
        print(f"Exception upgrading torch/torchvision: {e}")
        return None, None

def check_versions():
    global req_file
    reqs = open(req_file, 'r')
    lines = reqs.readlines()
    reqs_dict = {}
    for line in lines:
        splits = line.split("==")
        if len(splits) == 2:
            key = splits[0]
            reqs_dict[key] = splits[1].replace("\n", "").strip()
    reqs_dict["diffusers[torch]"] = "0.10.0"
    checks = ["bitsandbytes", "diffusers", "transformers"]

    xformers_ver = "0.0.17+48a77"
    torch_ver = "2.0.0.dev20230202+cu118"
    torch_vis_ver = "0.15.0.dev20230202+cu118"

    has_xformers = importlib.util.find_spec("xformers") is not None
    xformers_check = str(importlib_metadata.version("xformers")) if has_xformers else None
    if xformers_check != xformers_ver:
        run(f'"{python}" -m {xformers_cmd}', f"Installing xformers {xformers_ver}.", "Couldn't install torch.")

    # torch check
    has_torch = importlib.util.find_spec("torch") is not None
    has_torch_vision = importlib.util.find_spec("torchvision") is not None

    torch_check = str(importlib_metadata.version("torch")) if has_torch else None
    torch_vision_check = str(importlib_metadata.version("torchvision")) if has_torch_vision else None

    if torch_check != torch_ver or torch_vision_check != torch_vis_ver:
        torch_check, torch_vision_check = fix_torch()

    for check, ver, module in [(torch_check, torch_ver, "torch"),
                               (torch_vision_check, torch_vis_ver, "torchvision"),
                               (xformers_check, xformers_ver, "xformers")]:
        if check != ver:
            if not check:
                print(f"[!] {module} NOT installed.")
            else:
                print(f"[!] {module} version {check} installed.")
        else:
            print(f"[+] {module} version {check} installed.")
        
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
        if not check_available and check != "xformers":
            status = "[!]"
            print(f"{status} {check} NOT installed.")
        else:
            print(f"{status} {check} version {check_ver} installed.")



base_dir = os.path.dirname(os.path.realpath(__file__))
revision = ""
app_revision = ""

try:
    repo = git.Repo(base_dir)
    revision = repo.rev_parse("HEAD")
    app_repo = git.Repo(os.path.join(base_dir, "..", ".."))
    app_revision = app_repo.rev_parse("HEAD")
except:
    pass

print("")
print("#######################################################################################################")
print("Initializing Dreambooth")
print("If submitting an issue on github, please provide the below text for debugging purposes:")
print("")
print(f"Python revision: {sys.version}")
print(f"Dreambooth revision: {revision}")
print(f"SD-WebUI revision: {app_revision}")
print("")
dreambooth_skip_install = os.environ.get('DREAMBOOTH_SKIP_INSTALL', False)

try:
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")
    if requirements_file == req_file:
        dreambooth_skip_install = True
except:
    pass

if not dreambooth_skip_install:
    name = "Dreambooth"
    run(f'"{sys.executable}" -m pip install -r "{req_file}"', f"Checking {name} requirements...",
        f"Couldn't install {name} requirements.")

# Check for "different" B&B Files and copy only if necessary
if os.name == "nt":
    try:
        python = sys.executable
        bnb_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bitsandbytes_windows")
        bnb_dest = os.path.join(sysconfig.get_paths()["purelib"], "bitsandbytes")
        filecmp.clear_cache()
        copied = False
        for file in os.listdir(bnb_src):
            src_file = os.path.join(bnb_src, file)
            if file == "main.py" or file == "paths.py":
                dest = os.path.join(bnb_dest, "cuda_setup")
            else:
                dest = bnb_dest
            dest_file = os.path.join(dest, file)
            shutil.copy2(src_file, dest)
    except:
        pass

check_versions()
print("")
print("#######################################################################################################")
try:
    from modules import safe
    safe.load = safe.unsafe_torch_load
    import torch
    torch.load = safe.unsafe_torch_load
except:
    pass