@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=
set "TORCH_COMMAND=pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116"
set "REQS_FILE=.\extensions\sd_dreambooth_extension\requirements.txt"

call webui.bat
