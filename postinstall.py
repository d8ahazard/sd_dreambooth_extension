import filecmp
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig

import git
import requests


def run(command, desc=None, errdesc=None, custom_env=None, live=True):
    if desc:
        print(desc)

    if live:
        result = subprocess.run(command, shell=True, env=custom_env or os.environ)
        if result.returncode:
            raise RuntimeError(
                f"{errdesc or 'Error running command'}. Command: {command} Error code: {result.returncode}")
        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                            env=custom_env or os.environ)

    if result.returncode:
        message = f"{errdesc or 'Error running command'}. Command: {command} Error code: {result.returncode}\n"
        message += f"stdout: {result.stdout.decode(encoding='utf8', errors='ignore') or '<empty>'}\n"
        message += f"stderr: {result.stderr.decode(encoding='utf8', errors='ignore') or '<empty>'}\n"
        raise RuntimeError(message)

    return result.stdout.decode(encoding='utf8', errors='ignore')


def actual_install():
    if os.environ.get("PUBLIC_KEY", None):
        print("Docker, returning.")
        try:
            from extensions.sd_dreambooth_extension.dreambooth import shared
        except:
            from dreambooth.dreambooth import shared  # noqa
        shared.launch_error = None
        return
    if sys.version_info < (3, 8):
        import importlib_metadata
    else:
        import importlib.metadata as importlib_metadata

    req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

    def install_torch(torch_command, use_torch2):
        try:
            install_cmd = f'"{python}" -m {torch_command}'
            print(f"Torch install command: {install_cmd}")
            run(install_cmd, f"Installing torch{'2' if use_torch2 else ''} and torchvision.", "Couldn't install torch.")
            has_torch = importlib.util.find_spec("torch") is not None
            has_torch_vision = importlib.util.find_spec("torchvision") is not None
            if use_torch2:
                run(f"{python} -m pip install sympy==1.11.1")
            torch_check = str(importlib_metadata.version("torch")) if has_torch else None
            torch_vision_check = str(importlib_metadata.version("torchvision")) if has_torch_vision else None
            return torch_check, torch_vision_check
        except Exception as e:
            print(f"Exception upgrading torch/torchvision: {e}")
            return None, None

    def set_torch2_paths():
        # Get the URL for the latest release
        url = "https://github.com/ArrowM/xformers/releases/latest"
        response = requests.get(url)
        resolved_url = response.url
        last_portion = resolved_url.split("/")[-1]
        d_index = last_portion.index('.d')
        revisions = last_portion[d_index + 2:]
        revisions = revisions.split("-")
        if len(revisions) != 3:
            print("Unable to parse revision information.")
            return None
        torch_version = revisions[0]
        python_version = revisions[1]
        cuda_version = revisions[2]
        xformers_ver = last_portion.replace(f"-{python_version}-{cuda_version}", "")
        os_string = "win_amd64" if os.name == "nt" else "linux_x86_64"
        torch_ver = f"2.0.0.dev{torch_version}+{cuda_version}"
        torch_vis_ver = f"0.15.0.dev{torch_version}+{cuda_version}"
        xformers_url = f"{resolved_url}/{xformers_ver}-{python_version}-{python_version}-{os_string}.whl".replace(
            "/tag/", "/download/")
        torch2_url = f"https://download.pytorch.org/whl/nightly/{cuda_version}/torch-2.0.0.dev{torch_version}%2B{cuda_version}-{python_version}-{python_version}-{os_string}.whl"
        torchvision2_url = f"https://download.pytorch.org/whl/nightly/{cuda_version}/torchvision-0.15.0.dev{torch_version}%2B{cuda_version}-{python_version}-{python_version}-{os_string}.whl"
        triton_url = f"https://download.pytorch.org/whl/nightly/{cuda_version}/pytorch_triton-2.0.0%2B0d7e753227-{python_version}-{python_version}-linux_x86_64.whl"
        xformers_ver = xformers_ver.replace("xformers-", "")
        print(f"Xformers version: {xformers_ver}")
        print(f"Torch version: {torch_ver}")
        print(f"Torch vision version: {torch_vis_ver}")
        print(f"xu: {xformers_url}")
        print(f"tu: {torch2_url}")
        print(f"tvu: {torchvision2_url}")
        print(f"tru: {triton_url}")
        torch_final = f"{torch2_url} {torchvision2_url}"
        if os.name != "nt":
            torch_final += f" {triton_url}"
        return xformers_ver, torch_ver, torch_vis_ver, xformers_url, torch_final

    def check_versions():
        launch_errors = []
        use_torch2 = False
        # try:
        #     print(f"ARGS: {sys.argv}")
        #     if "--torch2" in sys.argv:
        #         use_torch2 = True
        #
        #     print(f"Torch2 Selected: {use_torch2}")
        # except:
        #     pass
        #
        # if use_torch2 and not (platform.system() == "Linux" or platform.system() == "Windows"):
        #     print(f"Xformers libraries for Torch2 are not available for {platform.system()} yet, disabling.")
        #     use_torch2 = False

        requirements = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
        # Open requirements file and read lines
        with open(requirements, 'r') as f:
            lines = f.readlines()

        # Create dictionary to store package names and version numbers
        reqs_dict = {}

        # Regular expression to match package names and version numbers
        pattern = r'^(?P<package_name>[\w-]+)(\[(?P<extras>[\w\s,-]+)\])?((?P<operator>==|~=)(?P<version>(\d+\.)*\d+([ab]\d+)?)(\.\w+)?(\.\w+(-\d+)?)?)?$'

        # Loop through each line in the requirements file
        for line in lines:
            # Strip whitespace and comments
            line = line.strip()
            if line.startswith('#'):
                continue
            # Use regular expression to extract package name and version number
            match = re.match(pattern, line)
            if match:
                package_name = match.group('package_name')
                version = match.group('version')
                # Split version number into three integers
                if version:
                    version_list = version.split('.')[:3]
                    # Remove any non-digit characters from the third version value
                    version_list[2] = ''.join(filter(str.isdigit, version_list[2]))
                    version_tuple = tuple(map(int, version_list))
                else:
                    version_tuple = None
                # Add package name and version tuple to dictionary
                reqs_dict[package_name] = version_tuple

        checks = ["bitsandbytes", "diffusers", "transformers", "xformers"]
        torch_ver = "1.13.1+cu117"
        torch_vis_ver = "0.14.1+cu117"
        xformers_ver = "0.0.17.dev464"

        # if use_torch2:
        #     xformers_ver, torch_ver, torch_vis_ver, xformers_url, torch_final = set_torch2_paths()
        #     print("Setting torch2 vars...")
        #     torch_cmd = f"pip install --no-deps {torch_final}"
        #     xformers_cmd = f"pip install {xformers_url}"
        #
        # else:
        #     xformers_ver = "0.0.17.dev447"
        #     torch_ver = "1.13.1+cu117"
        #     torch_vis_ver = "0.14.1+cu117"
        #     torch_cmd = "pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117"
        #     xformers_cmd = "pip install xformers==0.0.17.dev464"

        # # Check/install xformers
        # has_xformers = importlib.util.find_spec("xformers") is not None
        # xformers_check = str(importlib_metadata.version("xformers")) if has_xformers else None

        # if xformers_check != xformers_ver:
        #     run(f'"{python}" -m {xformers_cmd}', f"Installing xformers {xformers_ver} from {'pypi' if '==' in xformers_cmd else 'github'}.", "Couldn't install torch.")

        # torch check

        has_torch = importlib.util.find_spec("torch") is not None
        has_torch_vision = importlib.util.find_spec("torchvision") is not None
        # has_xformers = importlib.util.find_spec("xformers") is not None

        torch_check = str(importlib_metadata.version("torch")) if has_torch else None
        torch_vision_check = str(importlib_metadata.version("torchvision")) if has_torch_vision else None
        # xformers_check = str(importlib_metadata.version("xformers")) if has_xformers else None

        # if torch_check != torch_ver or torch_vision_check != torch_vis_ver:
        #     torch_ver, torch_vis_ver = install_torch(torch_cmd, use_torch2)

        for check, ver, module in [(torch_check, torch_ver, "torch"),
                                   (torch_vision_check, torch_vis_ver, "torchvision")]:

            if check != ver:
                if not check:
                    print(f"[!] {module} NOT installed.")
                    launch_errors.append(f"{module} not installed.")

                else:
                    print(f"[!] {module} version {check} installed.")
                    launch_errors.append(f"Incorrect version of {module} installed.")
            else:
                print(f"[+] {module} version {check} installed.")

        # Loop through each required package and check if it is installed
        for check in checks:
            check_ver = "N/A"
            status = "[ ]"
            try:
                check_available = importlib.util.find_spec(check) is not None
                if check_available:
                    check_ver = importlib_metadata.version(check)
                    check_version = tuple(map(int, re.split(r"[\.\+]", check_ver)[:3]))

                    if check in reqs_dict:
                        req_version = reqs_dict[check]
                        if req_version is None or check_version >= req_version:
                            status = "[+]"
                        else:
                            status = "[!]"
                            launch_errors.append(f"Incorrect version of {check} installed.")


            except importlib_metadata.PackageNotFoundError:
                print(f"No package for {check}")
                check_available = False
            if not check_available:
                status = "[!]"
                print(f"{status} {check} NOT installed.")
                launch_errors.append(f"{check} not installed.")
            else:
                print(f"{status} {check} version {check_ver} installed.")

        try:
            from modules.shared import cmd_opts
            xformers_flag = cmd_opts["xformers"]
            if not xformers_flag:
                error = "XFORMERS FLAG IS DISABLED, XFORMERS MUST BE ENABLED IN AUTO1111!"
                print(error)
                launch_errors.append(error)
        except:
            pass

        if len(launch_errors):
            print(f"Launch errors detected: {launch_errors}")
            os.environ["ERRORS"] = json.dumps(launch_errors)
        else:
            os.environ["ERRORS"] = ""

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

    python = sys.executable

    # Check for "different" B&B Files and copy only if necessary
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
