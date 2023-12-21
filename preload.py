import argparse
import logging
import os

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(name)s] - %(message)s', level=logging.DEBUG)
logger = logging.getLogger("launch")
# Set up logging
to_skip = ["urllib3", "PIL", "accelerate", "matplotlib", "h5py", "xformers", "tensorflow", "passlib", "asyncio",
           "tensorboard", "diffusers", "httpx"]
for skip in to_skip:
    logging.getLogger(skip).setLevel(logging.WARNING)


def preload(parser: argparse.ArgumentParser):
    # from postinstall import actual_install
    if os.name == "posix":
        # For now disable Torch2 Dynamo
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
    parser.add_argument("--dreambooth-models-path", type=str, help="Path to directory to store Dreambooth model file("
                                                                   "s).", default=None)
    parser.add_argument("--lora-models-path", type=str, help="Path to directory to store Lora model file(s).",
                        default=None)
    parser.add_argument("--ckptfix", action='store_true',
                        help="(Dreambooth) Enable fix for OOM errors when extracting checkpoints.")
    parser.add_argument("--force-cpu", action='store_true',
                        help="(Dreambooth) Train using CPU only.")
    parser.add_argument("--profile-db", action='store_true',
                        help="Set this to enable memory profiling while training. For science only.")
    parser.add_argument("--debug-db", action='store_true',
                        help="Set this to enable memory logging. For science only.")
