@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=
:: Use the below argument if getting OOM extracting checkpoints
:: set COMMANDLINE_ARGS=--ckptfix
set "TORCH_COMMAND=pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116"
set "REQS_FILE=.\extensions\sd_dreambooth_extension\requirements.txt"
:: Uncomment below to skip trying to install automatically on launch.
:: set "DREAMBOOTH_SKIP_INSTALL=True"
:: Use this to launch with accelerate (Run 'accelerate config' first, launch once without to install dependencies)
:: set ACCELERATE="True"

call webui.bat
