import argparse
import os

from modules.paths import models_path


def preload(parser: argparse.ArgumentParser):
    print("Preloading Dreambooth!")
    parser.add_argument("--dreambooth-models-path", type=str, help="Path to directory to store Dreambooth model file(s).", default=os.path.join(models_path, 'Dreambooth'))
