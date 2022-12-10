import argparse
import modules.safe
from extensions.sd_dreambooth_extension import reallysafe


def preload(parser: argparse.ArgumentParser):
    # We really shouldn't have to do this...
    modules.safe.RestrictedUnpickler = reallysafe.RestrictedUnpickler
    parser.add_argument("--dreambooth-models-path", type=str, help="Path to directory to store Dreambooth model file("
                                                                   "s).", default=None)
    parser.add_argument("--lora-models-path", type=str, help="Path to directory to store Lora model file(s).",
                        default=None)
    parser.add_argument("--test-lora", action='store_true', help="Enable this to test LORA. You should probably only do this if I told you about it.")
    parser.add_argument("--ckptfix", action='store_true', help="(Dreambooth) Enable fix for OOM errors when extracting checkpoints.")
