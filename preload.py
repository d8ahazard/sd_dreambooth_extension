import argparse

import gradio

import modules.safe
from extensions.sd_dreambooth_extension import reallysafe
from extensions.sd_dreambooth_extension.dreambooth import xattention


def preload(parser: argparse.ArgumentParser):
    # We really shouldn't have to do this...
    # gradio.blocks.Blocks.process_api = xattention.process_api
    modules.safe.RestrictedUnpickler = reallysafe.RestrictedUnpickler
    parser.add_argument("--dreambooth-models-path", type=str, help="Path to directory to store Dreambooth model file("
                                                                   "s).", default=None)
    parser.add_argument("--lora-models-path", type=str, help="Path to directory to store Lora model file(s).",
                        default=None)
    parser.add_argument("--ckptfix", action='store_true',
                        help="(Dreambooth) Enable fix for OOM errors when extracting checkpoints.")
    parser.add_argument("--profile-db", action='store_true',
                        help="Set this to enable memory profiling while training. For science only.")
    parser.add_argument("--debug-db", action='store_true',
                        help="Set this to enable memory logging. For science only.")
