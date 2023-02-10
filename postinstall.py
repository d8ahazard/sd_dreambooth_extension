import filecmp
import importlib.util
import os
import platform
import shutil
import sys
import sysconfig

import git
import requests

from launch import run


def actual_install():
    if os.environ.get("PUBLIC_KEY", None):
        print("Docker, returning.")
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
        xformers_url = f"{resolved_url}/{xformers_ver}-{python_version}-{python_version}-{os_string}.whl".replace("/tag/", "/download/")
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
        try:
            print(f"ARGS: {sys.argv}")
            if "--torch2" in sys.argv:
                use_torch2 = True

            print(f"Torch2 Selected: {use_torch2}")
        except:
            pass

        if use_torch2 and not (platform.system() == "Linux" or platform.system() == "Windows"):
            print(f"Xformers libraries for Torch2 are not available for {platform.system()} yet, disabling.")
            use_torch2 = False

        requirements = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
        reqs = open(requirements, 'r')
        lines = reqs.readlines()
        reqs_dict = {}
        for line in lines:
            splits = line.split("==")
            if len(splits) == 2:
                key = splits[0]
                reqs_dict[key] = splits[1].replace("\n", "").strip()

        checks = ["bitsandbytes", "diffusers", "transformers"]

        if use_torch2:
            xformers_ver, torch_ver, torch_vis_ver, xformers_url, torch_final = set_torch2_paths()
            print("Setting torch2 vars...")
            torch_cmd = f"pip install --no-deps {torch_final}"
            xformers_cmd = f"pip install {xformers_url}"

        else:
            xformers_ver = "0.0.17.dev447"
            torch_ver = "1.13.1+cu117"
            torch_vis_ver = "0.14.1+cu117"
            torch_cmd = "pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117"
            xformers_cmd = "pip install xformers==0.0.17.dev447"


        # Check/install xformers
        has_xformers = importlib.util.find_spec("xformers") is not None
        xformers_check = str(importlib_metadata.version("xformers")) if has_xformers else None
        if xformers_check != xformers_ver:
            run(f'"{python}" -m {xformers_cmd}', f"Installing xformers {xformers_ver} from {'pypi' if '==' in xformers_cmd else 'github'}.", "Couldn't install torch.")

        # torch check
        has_torch = importlib.util.find_spec("torch") is not None
        has_torch_vision = importlib.util.find_spec("torchvision") is not None

        torch_check = str(importlib_metadata.version("torch")) if has_torch else None
        torch_vision_check = str(importlib_metadata.version("torchvision")) if has_torch_vision else None

        if torch_check != torch_ver or torch_vision_check != torch_vis_ver:
            torch_ver, torch_vis_ver = install_torch(torch_cmd, use_torch2)

        for check, ver, module in [(torch_check, torch_ver, "torch"),
                                   (torch_vision_check, torch_vis_ver, "torchvision"),
                                   (xformers_check, xformers_ver, "xformers")]:
            if check != ver:
                if not check:
                    print(f"[!] {module} NOT installed.")
                    launch_errors.append(f"{module} not installed.")

                else:
                    print(f"[!] {module} version {check} installed.")
                    launch_errors.append(f"Incorrect version of {module} installed.")
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
                            launch_errors.append(f"Incorrect version of {check} installed.")

            except importlib_metadata.PackageNotFoundError:
                check_available = False
            if not check_available:
                status = "[!]"
                print(f"{status} {check} NOT installed.")
                launch_errors.append(f"{check} not installed.")
            else:
                print(f"{status} {check} version {check_ver} installed.")

        from extensions.sd_dreambooth_extension.dreambooth import shared

        if len(launch_errors):
            shared.launch_error = launch_errors
        else:
            shared.launch_error = None

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