#!usr/bin/env python
# -*- coding: utf-8 -*-

import os, argparse, sys, yaml
from GANDLF.cli import copyrightMessage, split_data_and_save_csvs


def main():
    parser = argparse.ArgumentParser(
        prog="GANDLF_SplitCSV",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Split the data into training, validation, and testing sets and save them as csvs in the output directory.\n\n"
        + copyrightMessage,
    )
    parser.add_argument(
        "-i",
        "--inputCSV",
        metavar="",
        default=None,
        type=str,
        required=True,
        help="Input CSV file which contains the data to be split.",
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="",
        default=None,
        required=True,
        type=str,
        help="The GaNDLF config (in YAML) with the `nested_training` key specified to the folds needed.",
    )
    parser.add_argument(
        "-o",
        "--outputDir",
        metavar="",
        default=None,
        type=str,
        required=True,
        help="Output directory to save the split data.",
    )

    args = parser.parse_args()

    # check for required parameters - this is needed here to keep the cli clean
    for param_none_check in [args.inputCSV, args.outputDir, args.config]:
        if param_none_check is None:
            sys.exit("ERROR: Missing required parameter:", param_none_check)

    inputCSV = os.path.normpath(args.inputCSV)
    outputDir = os.path.normpath(args.outputDir)
    # initialize default
    config = {"nested_training": {"testing": 5, "validation": 5}}
    if os.path.isfile(args.config):
        config = yaml.safe_load(open(args.config, "r"))

    print("Config used for split:", config)

    split_data_and_save_csvs(inputCSV, outputDir, config)

    print("Finished successfully.")


# main function
if __name__ == "__main__":
    main()
