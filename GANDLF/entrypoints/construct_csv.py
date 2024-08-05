#!usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import argparse
import ast
from typing import Optional
import yaml
import click
from deprecated import deprecated

from GANDLF.entrypoints import append_copyright_to_help
from GANDLF.utils import writeTrainingCSV

from GANDLF.cli import copyrightMessage
from GANDLF.utils import logger_setup


def _construct_csv(
    input_dir: str,
    channels_id: str,
    label_id: Optional[str],
    output_file: str,
    relativize_paths_to_output: bool,
):
    input_dir = os.path.normpath(input_dir)
    output_file = os.path.normpath(output_file)

    # Do some special handling for if users pass a yml file for channel/label IDs
    # This is used for MLCube functionality because MLCube does not support plain string inputs.
    if channels_id.endswith(".yml") or channels_id.endswith(".yaml"):
        if os.path.isfile(channels_id):
            with open(channels_id, "r") as f:
                content = yaml.safe_load(f)
                channels_id = content["channels"]
                if isinstance(channels_id, list):
                    channels_id = ",".join(channels_id)

                # TODO: raise a warning if label param is both passed as arg and defined in file
                if "label" in content:
                    label_id = content["label"]
                    if isinstance(label_id, list):  # TODO: it can be really a list?
                        label_id = ",".join(label_id)

    logging.debug(f"{input_dir=}")
    logging.debug(f"{channels_id=}")
    logging.debug(f"{label_id=}")
    logging.debug(f"{output_file=}")
    logging.debug(f"{relativize_paths_to_output=}")

    writeTrainingCSV(
        input_dir, channels_id, label_id, output_file, relativize_paths_to_output
    )


@click.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Input data directory which contains images in specified format",
)
@click.option(
    "--channels-id",
    "-c",
    required=True,
    help="Channels/modalities identifier string to check for in all files in 'input_dir'; for example: "
    "--channels-id _t1.nii.gz,_t2.nii.gz. May be a YAML file with `channels` list of suffixes",
    type=str,
)
@click.option(
    "--label-id",
    "-l",
    type=str,
    help="Label/mask identifier string to check for in all files in 'input_dir'; for example: "
    "--label-id _seg.nii.gz. Param value is ignored in `label` is defined in channels YAML file",
)
@click.option(
    "--output-file",
    "-o",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False),
    help="Output CSV file",
)
@click.option(
    "--relativize-paths",
    "-r",
    is_flag=True,
    help="If True, paths in the output data CSV will always be relative to the location"
    " of the output data CSV itself.",
)
@click.option(
    "--log-file",
    type=click.Path(),
    default=None,
    help="Output file which will contain the logs.",
)
@append_copyright_to_help
def new_way(
    input_dir: str,
    channels_id: str,
    label_id: Optional[str],
    output_file: str,
    relativize_paths: bool,
    log_file: str,
):
    """Generate training/inference CSV from data directory."""

    logger_setup(log_file)
    _construct_csv(
        input_dir=input_dir,
        channels_id=channels_id,
        label_id=label_id,
        output_file=output_file,
        relativize_paths_to_output=relativize_paths,
    )


@deprecated(
    "This is a deprecated way of running GanDLF. Please, use `gandlf construct-csv` cli command "
    + "instead of `gandlf_constructCSV`. Note that in new CLI tool some params were renamed:\n"
    + "  --inputDir to --input-dir\n"
    + "  --channelsID to --channels-id\n"
    + "  --labelID to --label-id\n"
    + "  --outputFile to --output-file\n"
    + "  --relativizePaths to --relativize-paths and converted to flag, i.e. no value required\n"
    + "`gandlf_constructCSV` script would be deprecated soon."
)
def old_way():
    logger_setup()
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
        "-o", "--outputFile", metavar="", type=str, help="Output CSV file"
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
    for param_name in ["inputDir", "channelsID", "outputFile"]:
        param_none_check = getattr(args, param_name)
        assert param_none_check is not None, f"Missing required parameter: {param_name}"

    _construct_csv(
        input_dir=args.inputDir,
        channels_id=args.channelsID,
        label_id=args.labelID,
        output_file=args.outputFile,
        relativize_paths_to_output=args.relativizePaths,
    )


# main function
if __name__ == "__main__":
    old_way()
