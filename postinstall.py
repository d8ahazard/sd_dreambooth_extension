import filecmp
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import traceback
from dataclasses import dataclass

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

    print("")
    print("Initializing Dreambooth")
    print(f"Dreambooth revision: {revision}")
    print("If submitting an issue on github, please provide the full startup log for debugging purposes.")
    print("")

    install_requirements()

    check_torch()

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
        print("pip_install Exception:")
        print("[Args]========================")
        print(list(args))
        print("[Message]=======================")
        try:
            error_msg = grepexc.stdout.decode()
            error_msg = [line for line in error_msg.split('\n') if line.strip()]
            print(error_msg)
        except Exception as e:
            print(e)
        print("===============================")
        try:
            error_msg = grepexc.stderr.decode()
            error_msg = [line for line in error_msg.split('\n') if line.strip()]
            print(error_msg)
        except Exception as e:
            print(e)
        print("===============================")
        try:
            error_msg = grepexc.output.decode()
            error_msg = [line for line in error_msg.split('\n') if line.strip()]
            print(error_msg)
        except Exception as e:
            print(e)
        print("===============================")


def install_requirements():
    dreambooth_skip_install = os.environ.get("DREAMBOOTH_SKIP_INSTALL", False)
    req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
    req_file_startup_arg = os.environ.get("REQS_FILE", "requirements_versions.txt")

    if dreambooth_skip_install or req_file == req_file_startup_arg:
        return

    pip_install("-r", req_file)


def check_torch():
    torch2_install = os.environ.get("TORCH2", False)
    if torch2_install:
        # torch
        try:
            torch_version = importlib_metadata.version("torch")
            torch_outdated = Version(torch_version) < Version("2")
            if torch_outdated:
                pip_install("--pre", "--force-reinstall", "torch", "--index-url", "https://download.pytorch.org/whl/cu118")
        except:
            pass
        # torchvision
        try:
            torch_vision_version = importlib_metadata.version("torchvision")
            torch_outdated = Version(torch_vision_version) < Version("0.15")
            if torch_outdated:
                pip_install("--pre", "--force-reinstall", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118")
        except:
            pass
        # xformers
        try:
            xformers_version = importlib_metadata.version("xformers")
            xformers_outdated = Version(xformers_version) < Version("0.0.17.dev")
            if xformers_outdated:
                pip_install("--pre", "--force-reinstall", "--no-deps", "xformers")
        except:
            pass


def check_xformers():
    """
    Install xformers 0.0.17 if necessary
    """
    try:
        xformers_version = importlib_metadata.version("xformers")
        xformers_outdated = Version(xformers_version) < Version("0.0.17.dev")
        if xformers_outdated:
            torch_version = importlib_metadata.version("torch")
            is_torch_1 = Version(torch_version) < Version("2")
            if is_torch_1:
                pip_install("xformers==0.0.17.dev476")
            else:
                pip_install("xformers", "--pre")
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


@dataclass
class Dependency:
    module: str
    version: str
    version_comparison: str = "min"
    required: bool = True


def check_versions():
    dependencies = [
        Dependency(module="xformers", version="0.0.17.dev", required=False),
        Dependency(module="torch", version="1.13.1+cu116"),
        Dependency(module="torchvision", version="0.14.1+cu116"),
        Dependency(module="accelerate", version="0.17.1"),
        Dependency(module="diffusers", version="0.14.0"),
        Dependency(module="transformers", version="4.25.1"),
        Dependency(module="bitsandbytes",  version="0.35.4", version_comparison="exact"),
    ]

    launch_errors = []

    for dependency in dependencies:
        module = dependency.module

        has_module = importlib.util.find_spec(module) is not None
        installed_ver = str(importlib_metadata.version(module)) if has_module else None

        if not installed_ver:
            if module != "xformers":
                launch_errors.append(f"{module} not installed.")

            print(f"[!] {module} NOT installed.")
            continue

        required_version = dependency.version
        required_comparison = dependency.version_comparison

        if required_comparison == "min" and Version(installed_ver) < Version(required_version):
            if "xformers" == module:
                print_xformers_error()
            else:
                launch_errors.append(f"{module} is below the required {required_version} version.")
            print(f"[!] {module} version {installed_ver} installed.")

        elif required_comparison == "exact" and Version(installed_ver) != Version(required_version):
            launch_errors.append(f"{module} is not the required {required_version} version.")
            print(f"[!] {module} version {installed_ver} installed.")

        else:
            print(f"[+] {module} version {installed_ver} installed.")

    try:
        if len(launch_errors):
            print_launch_errors(launch_errors)
            os.environ["ERRORS"] = json.dumps(launch_errors)
        else:
            os.environ["ERRORS"] = ""
    except Exception as e:
        print(e)


def print_xformers_error():
    print()
    print("#######################################################################################################")
    print("#                                       XFORMERS ISSUE DETECTED                                       #")
    print("#######################################################################################################")
    print("#")
    print("# Dreambooth Extension was not able to update your xformers to a compatible version.")
    print("# xformers will not be available for Dreambooth.")
    print("#")
    print("#######################################################################################################")

def print_launch_errors(launch_errors):
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


def check_torch_unsafe_load():
    try:
        from modules import safe
        safe.load = safe.unsafe_torch_load
        import torch
        torch.load = safe.unsafe_torch_load
    except:
        pass
