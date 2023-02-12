@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=
set "XFORMERS_PACKAGE=xformers==0.0.17.dev447"
:: Use the below argument if getting OOM extracting checkpoints
:: set COMMANDLINE_ARGS=--ckptfix
set "REQS_FILE=.\extensions\sd_dreambooth_extension\requirements.txt"
:: Uncomment below to skip trying to install automatically on launch.
:: set "DREAMBOOTH_SKIP_INSTALL=True"
:: Use this to launch with accelerate (Run 'accelerate config' first, launch once without to install dependencies)
:: set ACCELERATE="True"

call webui.bat
