#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from GANDLF.cli.patch_extraction import patch_extraction

from GANDLF.cli import copyrightMessage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GANDLF_PatchMiner",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Construct patches from whole slide image(s).\n\n"
        + copyrightMessage,
    )

    parser.add_argument(
        "-i",
        "--input_CSV",
        dest="input_path",
        help="input path for the tissue",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        default=None,
        required=True,
        help="output path for the patches",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        dest="config",
        help="config (in YAML) for running the patch miner. Needs 'scale' and 'patch_size' to be defined, otherwise defaults to 16 and (256, 256), respectively.",
        required=False,
    )

    args = parser.parse_args()

    patch_extraction(args.input_path, args.output_path, args.config)

    print("Finished.")
