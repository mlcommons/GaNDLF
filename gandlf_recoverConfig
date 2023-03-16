#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from GANDLF.cli import copyrightMessage, recover_config
import pickle
import os, sys
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GANDLF_RecoverConfig",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Recovers a config file from a GaNDLF model. If used from within a deployed GaNDLF MLCube, attempts to extract the config from the embedded model.\n\n"
        + copyrightMessage,
    )

    parser.add_argument(
        "-m",
        "--modeldir",
        metavar="",
        default="",
        type=str,
        help="Path to the model directory.",
    )
    parser.add_argument(
        "-c",
        "--mlcube",
        metavar="",
        type=str,
        help="Pass this option to attempt to extract the config from the embedded model in a GaNDLF MLCube (if any). Only useful in that context.",
    )
    parser.add_argument(
        "-o",
        "--outputFile",
        metavar="",
        type=str,
        help="Path to an output file where the config will be written.",
    )

    args = parser.parse_args()

    if args.mlcube:
        search_dir = "/embedded_model/"
    else:
        search_dir = args.modeldir

    result = recover_config(search_dir, args.outputFile)
    assert result, "Config file recovery failed."
