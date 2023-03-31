import filecmp
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
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

    print("If submitting an issue on github, please provide the full startup log for debugging purposes.")
    print("")
    print("Initializing Dreambooth")
    print(f"Dreambooth revision: {revision}")

    install_requirements()

    check_xformers()

    check_bitsandbytes()

    check_versions()

    check_torch_unsafe_load()


def pip_install(*args):
    output = subprocess.check_output(
        [sys.executable, "-m", "pip", "install"] + list(args),
        stderr=subprocess.STDOUT,
        )
    for line in output.decode().split("\n"):
        if "Successfully installed" in line:
            print(line)


def install_requirements():
    dreambooth_skip_install = os.environ.get("DREAMBOOTH_SKIP_INSTALL", False)
    req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
    req_file_startup_arg = os.environ.get("REQS_FILE", "requirements_versions.txt")

    if dreambooth_skip_install or req_file == req_file_startup_arg:
        return

    has_diffusers = importlib.util.find_spec("diffusers") is not None
    has_tqdm = importlib.util.find_spec("tqdm") is not None
    transformers_version = importlib_metadata.version("transformers")

    try:
        pip_install("-r", req_file)

        if has_diffusers and has_tqdm and Version(transformers_version) < Version("4.26.1"):
            print()
            print("Does your project take forever to startup?")
            print("Repetitive dependency installation may be the reason.")
            print("Automatic1111's base project sets strict requirements on outdated dependencies.")
            print("If an extension is using a newer version, the dependency is uninstalled and reinstalled twice every startup.")
            print()
    except subprocess.CalledProcessError as grepexc:
        error_msg = grepexc.stdout.decode()
        print_requirement_installation_error(error_msg)
        raise grepexc


def check_xformers():
    """
    Install xformers if necessary
    """
    try:
        xformers_version = importlib_metadata.version("xformers")
        xformers_outdated = Version(xformers_version) < Version("0.0.17.dev")
        if xformers_outdated:
            try:
                torch_version = importlib_metadata.version("torch")
                is_torch_1 = Version(torch_version) < Version("2")
                if is_torch_1:
                    print_xformers_torch1_instructions(xformers_version)
                else:
                    pip_install("--force-reinstall", "xformers")
            except subprocess.CalledProcessError as grepexc:
                error_msg = grepexc.stdout.decode()
                print_xformers_installation_error(error_msg)
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
    import platform
    from sys import platform as sys_platform
    is_mac = sys_platform == 'darwin' and platform.machine() == 'arm64'

    dependencies = [
        Dependency(module="xformers", version="0.0.17.dev", required=False),
        Dependency(module="torch", version="1.13.1" if is_mac else "1.13.1+cu116"),
        Dependency(module="torchvision", version="0.14.1" if is_mac else "0.14.1+cu116"),
        Dependency(module="accelerate", version="0.17.1"),
        Dependency(module="diffusers", version="0.14.0"),
        Dependency(module="transformers", version="4.25.1"),
        Dependency(module="bitsandbytes",  version="0.35.4", version_comparison="exact"),
    ]

    launch_errors = []

    for dependency in dependencies:
        module = dependency.module

        has_module = importlib.util.find_spec(module) is not None
        installed_ver = importlib_metadata.version(module) if has_module else None

        if not installed_ver:
            if module != "xformers":
                launch_errors.append(f"{module} not installed.")

            print(f"[!] {module} NOT installed.")
            continue

        required_version = dependency.version
        required_comparison = dependency.version_comparison

        if required_comparison == "min" and Version(installed_ver) < Version(required_version):
            if dependency.required:
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


def print_requirement_installation_error(err):
    print("# Requirement installation exception:")
    for line in err.split('\n'):
        line = line.strip()
        if line:
            print(line)


def print_xformers_installation_error(err):
    torch_ver = importlib_metadata.version("torch")
    print()
    print("#######################################################################################################")
    print("#                                       XFORMERS ISSUE DETECTED                                       #")
    print("#######################################################################################################")
    print("#")
    print(f"# Dreambooth could not find a compatible version of xformers (>= 0.0.17.dev built with torch {torch_ver})")
    print("# xformers will not be available for Dreambooth. Consider upgrading to Torch 2.")
    print("#")
    print("# Xformers installation exception:")
    for line in err.split('\n'):
        line = line.strip()
        if line:
            print(line)
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


def print_xformers_torch1_instructions(xformers_version):
    print(f"# Your version of xformers is {xformers_version}.")
    print("# xformers >= 0.0.17.dev is required to be available on the Dreambooth tab.")
    print("# Torch 1 wheels of xformers >= 0.0.17.dev are no longer available on PyPI,")
    print("# but you can manually download them by going to:")
    print("https://github.com/facebookresearch/xformers/actions")
    print("# Click on the most recent action tagged with a release (middle column).")
    print("# Select a download based on your environment.")
    print("# Unzip your download")
    print("# Activate your venv and install the wheel: (from A1111 project root)")
    print("cd venv/Scripts")
    print("activate")
    print("pip install {REPLACE WITH PATH TO YOUR UNZIPPED .whl file}")
    print("# Then restart your project.")
