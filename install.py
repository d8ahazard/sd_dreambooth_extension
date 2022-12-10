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
    # print(f"Reqs dict: {reqs_dict}")
    reqs_dict["diffusers[torch]"] = "0.10.0"
    checks = ["bitsandbytes", "diffusers", "transformers", "xformers", "torch", "torchvision"]
    flat_check = ["xformers", "torch", "torchvision"]
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
                if check in flat_check:
                    status = "[+]"
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
print("#######################################################################################################")
