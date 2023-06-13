#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from GANDLF.cli import copyrightMessage, post_training_model_optimization


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GANDLF_OptimizeModel",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate optimized versions of trained GaNDLF models.\n\n"
        + copyrightMessage,
    )

    parser.add_argument(
        "-m",
        "--model",
        metavar="",
        type=str,
        help="Path to the model file (ending in '.pth.tar') you wish to optimize.",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="",
        type=str,
        default=None,
        required=False,
        help="The configuration file (contains all the information related to the training/inference session).",
    )

    args = parser.parse_args()

    if post_training_model_optimization(args.model, args.config):
        print("Post-training model optimization successful.")
    else:
        print("Post-training model optimization failed.")
