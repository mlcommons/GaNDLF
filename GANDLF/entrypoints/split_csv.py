#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import yaml
from typing import Optional

import click
from deprecated.classic import deprecated

from GANDLF.cli import copyrightMessage, split_data_and_save_csvs
from GANDLF.entrypoints import append_copyright_to_help


def _split_csv(input_csv: str, output_dir: str, config_path: Optional[str]):
    input_csv = os.path.normpath(input_csv)
    output_dir = os.path.normpath(output_dir)
    # initialize default
    config = yaml.safe_load(open(config_path, "r"))

    print("Config used for split:", config)

    split_data_and_save_csvs(input_csv, output_dir, config)

    print("Finished successfully.")


@click.command()
@click.option(
    "--input-csv",
    "-i",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Input CSV file which contains the data to be split.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Output directory to save the split data.",
)
@click.option(
    "--config",
    "-c",
    required=True,
    help="The GaNDLF config (in YAML) with the `nested_training` key specified to the folds needed.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@append_copyright_to_help
def new_way(input_csv: str, output_dir: str, config: Optional[str]):
    """Split the data into training, validation, and testing sets and save them as csvs in the output directory."""
    _split_csv(input_csv, output_dir, config)


# old-fashioned way of running gandlf via `gandlf_splitCSV`.
@deprecated(
    "This is a deprecated way of running GanDLF. Please, use `gandlf split-csv` cli command "
    + "instead of `gandlf_splitCSV`. Note that in new CLI tool some params were renamed:\n"
    + "  --inputCSV to --input-csv\n"
    + "  --outputDir to --output-dir\n"
    + "`gandlf_splitCSV` script would be deprecated soon."
)
def old_way():
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
    for param_name in ["inputCSV", "outputDir", "config"]:
        param_none_check = getattr(args, param_name)
        if param_none_check is None:
            sys.exit(f"ERROR: Missing required parameter: {param_name}")

    _split_csv(
        input_csv=args.inputCSV, output_dir=args.outputDir, config_path=args.config
    )


# main function
if __name__ == "__main__":
    old_way()
