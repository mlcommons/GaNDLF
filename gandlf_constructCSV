#!usr/bin/env python
# -*- coding: utf-8 -*-

import os, argparse, sys, ast
from datetime import date
from GANDLF.utils import writeTrainingCSV

from GANDLF.cli import copyrightMessage

import yaml


def main():
    parser = argparse.ArgumentParser(
        prog="GANDLF_ConstructCSV",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate training/inference CSV from data directory.\n\n"
        + copyrightMessage,
    )
    parser.add_argument(
        "-i",
        "--inputDir",
        metavar="",
        type=str,
        help="Input data directory which contains images in specified format",
    )
    parser.add_argument(
        "-c",
        "--channelsID",
        metavar="",
        type=str,
        help="Channels/modalities identifier string to check for in all files in 'input_dir'; for example: --channelsID _t1.nii.gz,_t2.nii.gz",
    )
    parser.add_argument(
        "-l",
        "--labelID",
        default=None,
        type=str,
        help="Label/mask identifier string to check for in all files in 'input_dir'; for example: --labelID _seg.nii.gz",
    )
    parser.add_argument(
        "-o",
        "--outputFile",
        metavar="",
        type=str,
        help="Output CSV file",
    )
    parser.add_argument(
        "-r",
        "--relativizePaths",
        metavar="",
        type=ast.literal_eval,
        default=False,
        help="If True, paths in the output data CSV will always be relative to the location of the output data CSV itself.",
    )

    args = parser.parse_args()

    # check for required parameters - this is needed here to keep the cli clean
    for param_none_check in [
        args.inputDir,
        args.channelsID,
        args.outputFile,
    ]:
        if param_none_check is None:
            sys.exit("ERROR: Missing required parameter:", param_none_check)

    inputDir = os.path.normpath(args.inputDir)
    outputFile = os.path.normpath(args.outputFile)
    channelsID = args.channelsID
    labelID = args.labelID
    relativizePathsToOutput = args.relativizePaths

    # Do some special handling for if users pass a yml file for channel/label IDs
    # This is used for MLCube functionality because MLCube does not support plain string inputs.
    if channelsID.endswith(".yml") or channelsID.endswith(".yaml"):
        if os.path.isfile(channelsID):
            with open(channelsID, "r") as f:
                content = yaml.safe_load(f)
                channelsID = content["channels"]
                if isinstance(channelsID, list):
                    channelsID = ",".join(channelsID)

                if "label" in content:
                    labelID = content["label"]
                    if isinstance(labelID, list):
                        channelsID = ",".join(channelsID)

    writeTrainingCSV(inputDir, channelsID, labelID, outputFile, relativizePathsToOutput)


# main function
if __name__ == "__main__":
    main()
