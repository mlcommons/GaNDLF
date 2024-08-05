#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import click
from deprecated import deprecated
from typing import Optional

from GANDLF import version
from GANDLF.cli import copyrightMessage
from GANDLF.cli.generate_metrics import generate_metrics_dict
from GANDLF.entrypoints import append_copyright_to_help
from GANDLF.utils import logger_setup


def _generate_metrics(
    input_data: str,
    config: str,
    output_file: Optional[str],
    missing_prediction: int = -1,
):
    generate_metrics_dict(input_data, config, output_file, missing_prediction)
    print("Finished.")


@click.command()
@click.option(
    "--config",
    "-c",
    required=True,
    help="The configuration file (contains all the information related to the training/inference session)",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--input-data",
    "-i",
    required=True,
    type=str,
    help="The CSV file of input data that is used to generate the metrics; "
    "should contain 3 columns: 'SubjectID,Target,Prediction'",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(file_okay=True, dir_okay=False),
    help="Location to save the output dictionary. If not provided, will print to stdout.",
)
@click.option(
    "--missing-prediction",
    "-m",
    required=False,
    type=int,
    default=-1,
    help="The value to use for missing predictions as penalty; if `-1`, this does not get added. This is only used in the case where the targets and predictions are passed independently.",
)
@click.option(
    "--log-file",
    type=click.Path(),
    default=None,
    help="Output file which will contain the logs.",
)
@click.option("--raw-input", hidden=True)
@append_copyright_to_help
def new_way(
    config: str,
    input_data: str,
    output_file: Optional[str],
    missing_prediction: int,
    raw_input: str,
    log_file: str,
):
    """Metrics calculator."""

    logger_setup(log_file)
    _generate_metrics(
        input_data=input_data,
        config=config,
        output_file=output_file,
        missing_prediction=missing_prediction,
    )


@deprecated(
    "This is a deprecated way of running GanDLF. Please, use `gandlf generate-metrics` cli command "
    + "instead of `gandlf_generateMetrics`. Note that in new CLI tool some params were renamed or "
    "changed its behavior:\n"
    + "  --parameters_file to --config\n"
    + "  --inputdata/--data_path to --input-data\n"
    + "  --outputfile/--output_path to --output-file\n"
    + "  --version removed; use `gandlf --version` instead\n"
    + "`gandlf_generateMetrics` script would be deprecated soon."
)
def old_way():
    logger_setup()
    parser = argparse.ArgumentParser(
        prog="GANDLF_Metrics",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Metrics calculator.\n\n" + copyrightMessage,
    )
    parser.add_argument(
        "-c",
        "--config",
        "--parameters_file",
        metavar="",
        type=str,
        required=True,
        help="The configuration file (contains all the information related to the training/inference session)",
    )
    parser.add_argument(
        "-i",
        "--inputdata",
        "--data_path",
        metavar="",
        type=str,
        required=True,
        help="The CSV file of input data that is used to generate the metrics; should contain 3 columns: 'SubjectID,Target,Prediction'",
    )
    parser.add_argument(
        "-o",
        "--outputfile",
        "--output_path",
        metavar="",
        type=str,
        default=None,
        help="Location to save the output dictionary. If not provided, will print to stdout.",
    )
    parser.add_argument(
        "-m",
        "--missingprediction",
        metavar="",
        type=int,
        default=-1,
        help="The value to use for missing predictions as penalty; if `-1`, this does not get added. This is only used in the case where the targets and predictions are passed independently.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s v{}".format(version) + "\n\n" + copyrightMessage,
        help="Show program's version number and exit.",
    )

    # This is a dummy argument that exists to trigger MLCube mounting requirements.
    # Do not remove.
    parser.add_argument("-rawinput", "--rawinput", help=argparse.SUPPRESS)

    args = parser.parse_args()
    assert args.config is not None, "Missing required parameter: config"
    assert args.inputdata is not None, "Missing required parameter: inputdata"

    _generate_metrics(
        input_data=args.inputdata,
        config=args.config,
        output_file=args.outputfile,
        missing_prediction=args.missingprediction,
    )


if __name__ == "__main__":
    old_way()
