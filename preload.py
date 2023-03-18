import argparse


def preload(parser: argparse.ArgumentParser):
    # from postinstall import actual_install

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
    # parser.add_argument("--torch2", action='store_true',
    #                     help="Enable this flag to use torch V2.")
    #
    # actual_install()
