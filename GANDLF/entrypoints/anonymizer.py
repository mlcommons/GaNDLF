#!usr/bin/env python
# -*- coding: utf-8 -*-

import os, argparse, sys, yaml
from GANDLF.anonymize import run_anonymizer
from GANDLF.cli import copyrightMessage


def main():
    parser = argparse.ArgumentParser(
        prog="GANDLF_Anonymize",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Anonymize images/scans in the data directory.\n\n"
        + copyrightMessage,
    )
    parser.add_argument(
        "-i",
        "--inputDir",
        metavar="",
        type=str,
        help="Input directory or file which contains images to be anonymized.",
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="",
        default="",
        type=str,
        help="config (in YAML) for running anonymization, optionally, specify modality using '-m' for defaults.",
    )
    parser.add_argument(
        "-m",
        "--modality",
        metavar="",
        default="rad",
        type=str,
        help="The modality type, can be 'rad' or 'histo'.",
    )
    parser.add_argument(
        "-o",
        "--outputFile",
        metavar="",
        type=str,
        help="Output directory or file which will contain the image(s) after anonymization.",
    )

    args = parser.parse_args()

    # check for required parameters - this is needed here to keep the cli clean
    for param_none_check in [args.inputDir, args.outputFile]:
        if param_none_check is None:
            sys.exit("ERROR: Missing required parameter:", param_none_check)

    inputDir = os.path.normpath(args.inputDir)
    outputFile = os.path.normpath(args.outputFile)
    if os.path.isfile(args.config):
        config = yaml.safe_load(open(args.config, "r"))
    else:
        config = None

    run_anonymizer(inputDir, outputFile, config, args.modality)

    print("Finished successfully.")


# main function
if __name__ == "__main__":
    main()
