import shutil
from launch import run_pip
import os
from modules.paths import script_path

reqs = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
run_pip(f"install -r {reqs}", "requirements for Dreambooth")

if os.name == "nt":
    bnb_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bitsandbytes_windows")
    site_dir = os.path.join(script_path, "venv", "Lib", "site-packages", "bitsandbytes")
    if not os.path.exists(os.path.join(site_dir, "libbitsandbytes_cpu.dll")):
        print("Copying 8Bit Adam files for Windows.")
        for file in os.listdir(bnb_dir):
            fullfile = os.path.join(bnb_dir, file)
            shutil.copy(fullfile, site_dir)
