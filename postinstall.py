import filecmp
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from typing import List

import git
from packaging.version import Version

from dreambooth import shared

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


def actual_install():
    if os.environ.get("PUBLIC_KEY", None):
        print("Docker, returning.")
        shared.launch_error = None
        return

    base_dir = os.path.dirname(os.path.realpath(__file__))

    try:
        repo = git.Repo(base_dir)
        revision = repo.rev_parse("HEAD")
    except:
        revision = ""

    try:
        app_repo = git.Repo(os.path.join(base_dir, "../.."))
        app_revision = app_repo.rev_parse("HEAD")
    except:
        app_revision = ""

    print("")
    print("Initializing Dreambooth")
    print("If submitting an issue on github, please provide the below text for debugging purposes:")
    print("")
    print(f"Python revision: {sys.version}")
    print(f"Dreambooth revision: {revision}")
    print(f"SD-WebUI revision: {app_revision}")
    print("")

    install_requirements()

    check_xformers()

    check_bitsandbytes()

    check_versions()

    check_torch_unsafe_load()


def pip_install(*args):
    try:
        output = subprocess.check_output(
                [sys.executable, "-m", "pip", "install"] + list(args),
                stderr=subprocess.STDOUT,
            )
        for line in output.decode().split("\n"):
            if "Successfully installed" in line:
                print(line)
    except subprocess.CalledProcessError as grepexc:
        error_msg = grepexc.output.decode()
        print(error_msg)


def install_requirements():
    dreambooth_skip_install = os.environ.get("DREAMBOOTH_SKIP_INSTALL", False)
    req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
    req_file_startup_arg = os.environ.get("REQS_FILE", "requirements_versions.txt")

    if dreambooth_skip_install or req_file == req_file_startup_arg:
        return

    # Necessary for the loop below
    import platform
    platform_machine = platform.machine()
    from sys import platform as sys_platform
    lines = open(req_file, "r").read().split("\n")

    for line in lines:
        if ";" in line:
            [lib, cond] = line.split(";")
            if not eval(cond):
                continue
        else:
            lib = line

        pip_install(lib)

    print()


def check_xformers():
    """
    Install xformers 0.0.17 if necessary
    """
    try:
        xformers_version = importlib_metadata.version("xformers")
        is_xformers_outdated = Version(xformers_version) < Version("0.0.17.dev")
        if is_xformers_outdated:
            pip_install("--no-deps", "xformers==0.0.17.dev476")
            pip_install("numpy")
            pip_install("pyre-extensions")
    except:
        pass


def check_bitsandbytes():
    """
    Check for "different" B&B Files and copy only if necessary
    """
    if os.name == "nt":
        try:
            bnb_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bitsandbytes_windows")
            bnb_dest = os.path.join(sysconfig.get_paths()["purelib"], "bitsandbytes")
            filecmp.clear_cache()
            for file in os.listdir(bnb_src):
                src_file = os.path.join(bnb_src, file)
                if file == "main.py" or file == "paths.py":
                    dest = os.path.join(bnb_dest, "cuda_setup")
                else:
                    dest = bnb_dest
                shutil.copy2(src_file, dest)
        except:
            pass


def check_versions():
    required_libs = {
        "torch": ["min", "1.13.1+cu116"],
        "torchvision": ["min", "0.14.1+cu116"],
        "xformers": ["min", "0.0.17.dev"],
        "accelerate": ["min", "0.17.1"],
        "diffusers": ["min", "0.14.0"],
        "transformers": ["min", "4.25.1"],

        "bitsandbytes": ["exact", "0.35.4"],
    }

    launch_errors = []

    for module, req in required_libs.items():
        has_module = importlib.util.find_spec(module) is not None
        installed_ver = str(importlib_metadata.version(module)) if has_module else None

        if not installed_ver:
            if module != "xformers":
                launch_errors.append(f"{module} not installed.")

            print(f"[!] {module} NOT installed.")
            continue

        req_type, req_ver = req
        if req_type == "min" and Version(installed_ver) < Version(req_ver):
            print(f"[!] {module} version {installed_ver} installed.")
            launch_errors.append(f"{module} is below the required {req_ver} version.")

        elif req_type == "exact" and Version(installed_ver) != Version(req_ver):
            print(f"[!] {module} version {installed_ver} installed.")
            launch_errors.append(f"{module} is not the required {req_ver} version.")

        else:
            print(f"[+] {module} version {installed_ver} installed.")

    try:
        if len(launch_errors):
            print()
            print("#######################################################################################################")
            print("#                                       LIBRARY ISSUE DETECTED                                        #")
            print("#######################################################################################################")
            print("#")
            print("# " + "\n# ".join(launch_errors))
            print("#")
            print("# Dreambooth may not work properly.")
            print("#")
            print("# TROUBLESHOOTING")
            print("# 1. Fully restart your project (not just the webpage)")
            print("# 2. Update your A1111 project and extensions")
            print("# 3. Dreambooth requirements should have installed automatically, but you can manually install them")
            print("#    by running the following 4 commands from the A1111 project root:")
            print("cd venv/Scripts")
            print("activate")
            print("cd ../..")
            print("pip install -r ./extensions/sd_dreambooth_extension/requirements.txt")
            print("#######################################################################################################")

            os.environ["ERRORS"] = json.dumps(launch_errors)
        else:
            os.environ["ERRORS"] = ""
    except Exception as e:
        print(e)


def check_torch_unsafe_load():
    try:
        from modules import safe
        safe.load = safe.unsafe_torch_load
        import torch
        torch.load = safe.unsafe_torch_load
    except:
        pass


# def install_torch(torch_command, use_torch2):
#     try:
#         install_cmd = f""{python}" -m {torch_command}"
#         print(f"Torch install command: {install_cmd}")
#         run(install_cmd, f"Installing torch{"2" if use_torch2 else ""} and torchvision.", "Couldn"t install torch.")
#         has_torch = importlib.util.find_spec("torch") is not None
#         has_torch_vision = importlib.util.find_spec("torchvision") is not None
#         if use_torch2:
#             run(f"{python} -m pip install sympy==1.11.1")
#         torch_installed_ver = str(importlib_metadata.version("torch")) if has_torch else None
#         torch_vision_check = str(importlib_metadata.version("torchvision")) if has_torch_vision else None
#         return torch_installed_ver, torch_vision_check
#     except Exception as e:
#         print(f"Exception upgrading torch/torchvision: {e}")
#         return None, None
#
# def set_torch2_paths():
#     # Get the URL for the latest release
#     url = "https://github.com/ArrowM/xformers/releases/latest"
#     response = requests.get(url)
#     resolved_url = response.url
#     last_portion = resolved_url.split("/")[-1]
#     d_index = last_portion.index(".d")
#     revisions = last_portion[d_index + 2:]
#     revisions = revisions.split("-")
#     if len(revisions) != 3:
#         print("Unable to parse revision information.")
#         return None
#     torch_version = revisions[0]
#     python_version = revisions[1]
#     cuda_version = revisions[2]
#     xformers_ver = last_portion.replace(f"-{python_version}-{cuda_version}", "")
#     os_string = "win_amd64" if os.name == "nt" else "linux_x86_64"
#     torch_ver = f"2.0.0.dev{torch_version}+{cuda_version}"
#     torch_vis_ver = f"0.15.0.dev{torch_version}+{cuda_version}"
#     xformers_url = f"{resolved_url}/{xformers_ver}-{python_version}-{python_version}-{os_string}.whl".replace(
#         "/tag/", "/download/")
#     torch2_url = f"https://download.pytorch.org/whl/nightly/{cuda_version}/torch-2.0.0.dev{torch_version}%2B{cuda_version}-{python_version}-{python_version}-{os_string}.whl"
#     torchvision2_url = f"https://download.pytorch.org/whl/nightly/{cuda_version}/torchvision-0.15.0.dev{torch_version}%2B{cuda_version}-{python_version}-{python_version}-{os_string}.whl"
#     triton_url = f"https://download.pytorch.org/whl/nightly/{cuda_version}/pytorch_triton-2.0.0%2B0d7e753227-{python_version}-{python_version}-linux_x86_64.whl"
#     xformers_ver = xformers_ver.replace("xformers-", "")
#     print(f"Xformers version: {xformers_ver}")
#     print(f"Torch version: {torch_ver}")
#     print(f"Torch vision version: {torch_vis_ver}")
#     print(f"xu: {xformers_url}")
#     print(f"tu: {torch2_url}")
#     print(f"tvu: {torchvision2_url}")
#     print(f"tru: {triton_url}")
#     torch_final = f"{torch2_url} {torchvision2_url}"
#     if os.name != "nt":
#         torch_final += f" {triton_url}"
#     return xformers_ver, torch_ver, torch_vis_ver, xformers_url, torch_final
