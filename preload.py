import argparse
import os

from modules.paths import models_path


def preload(parser: argparse.ArgumentParser):
    parser.add_argument("--dreambooth-models-path", type=str, help="Path to directory to store Dreambooth model file("
                                                                   "s).", default=os.path.join(models_path,
                                                                                               'Dreambooth'))
    parser.add_argument("--ckptfix", action='store_true', help="(Dreambooth) Enable fix for OOM errors when extracting checkpoints.")
