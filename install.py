from launch import run_pip
import os
reqs = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
run_pip(f"install -r {reqs}", "requirements for Dreambooth")