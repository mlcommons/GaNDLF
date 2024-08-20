#!usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import argparse
import yaml
from typing import Optional
import click
from deprecated import deprecated

from GANDLF.anonymize import run_anonymizer
from GANDLF.cli import copyrightMessage
from GANDLF.entrypoints import append_copyright_to_help
from GANDLF.utils.gandlf_logging import logger_setup


def _anonymize_images(
    input_dir: str, output_file: str, config_path: Optional[str], modality: str
):
    input_dir = os.path.normpath(input_dir)
    output_file = os.path.normpath(output_file)
    # TODO: raise an error if config pass provided but not exist (user made a typo?)
    config = None
    if config_path and os.path.isfile(config_path):
        config = yaml.safe_load(open(config_path, "r"))

    logging.debug(f"{input_dir=}")
    logging.debug(f"{output_file=}")
    logging.debug(f"{config=}")
    logging.debug(f"{modality=}")
    run_anonymizer(input_dir, output_file, config, modality)

    logging.info("Finished successfully.")


# new way of defining params via click
@click.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input directory or file which contains images to be anonymized.",
)
@click.option(
    "--config",
    "-c",
    help="Config (in YAML) for running anonymization, optionally, specify modality using '-m' for defaults.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--modality",
    "-m",
    default="rad",
    type=click.Choice(["rad", "histo"]),
    help="The modality type, can be 'rad' or 'histo'.",
)
@click.option(
    "--output-file",
    "-o",
    required=True,
    type=click.Path(),
    help="Output directory or file which will contain the image(s) after anonymization.",
)
@click.option(
    "--log-file",
    type=click.Path(),
    default=None,
    help="Output file which will contain the logs.",
)
@append_copyright_to_help
def new_way(input_dir, config, modality, output_file, log_file):
    """Anonymize images/scans in the data directory."""
    logger_setup(log_file)
    _anonymize_images(input_dir, output_file, config, modality)


# old-fashioned way of running gandlf via `gandlf_anonymizer`.
@deprecated(
    "This is a deprecated way of running GanDLF. Please, use `gandlf anonymizer` cli command "
    + "instead of `gandlf_anonymizer`. Note that in new CLI tool some params were renamed:\n"
    + "  --inputDir to --input-dir\n"
    + "  --outputFile to --output-file\n"
    + "`gandlf_anonymizer` script would be deprecated soon."
)
def old_way():
    logger_setup()
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
    for param_name in ["inputDir", "outputFile"]:
        param_none_check = getattr(args, param_name)
        assert param_none_check is not None, f"Missing required parameter: {param_name}"

    inputDir = args.inputDir
    outputFile = args.outputFile
    config = args.config or None

    _anonymize_images(inputDir, outputFile, config, args.modality)


# main function
if __name__ == "__main__":
    old_way()
