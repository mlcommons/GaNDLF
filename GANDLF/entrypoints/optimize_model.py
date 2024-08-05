#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from typing import Optional

from deprecated import deprecated
import click

from GANDLF.cli import copyrightMessage, post_training_model_optimization
from GANDLF.entrypoints import append_copyright_to_help
from GANDLF.utils import logger_setup


def _optimize_model(
    model: str, config: Optional[str], output_path: Optional[str] = None
):
    if post_training_model_optimization(
        model_path=model, config_path=config, output_path=output_path
    ):
        print("Post-training model optimization successful.")
    else:
        print("Post-training model optimization failed.")


@click.command()
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to the model file (ending in '.pth.tar') you wish to optimize.",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
    help="Location to save the optimized model, defaults to location of `model`",
)
@click.option(
    "--config",
    "-c",
    help="The configuration file (contains all the information related to the training/inference session)."
    "Arg value is used if no config in model is found.",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--log-file",
    type=click.Path(),
    default=None,
    help="Output file which will contain the logs.",
)
@append_copyright_to_help
def new_way(
    model: str,
    log_file: str,
    config: Optional[str] = None,
    output_path: Optional[str] = None,
):
    """Generate optimized versions of trained GaNDLF models."""

    logger_setup(log_file)
    _optimize_model(model=model, config=config, output_path=output_path)


# old-fashioned way of running gandlf via `gandlf_optimizeModel`.
@deprecated(
    "This is a deprecated way of running GanDLF. Please, use `gandlf optimize-model` cli command "
    + "instead of `gandlf_optimizeModel`.\n"
    + "`gandlf_optimizeModel` script would be deprecated soon."
)
def old_way():
    logger_setup()
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
        "-o",
        "--outputdir",
        "--output_path",
        metavar="",
        type=str,
        default=None,
        help="Location to save the optimized model, defaults to location of `model`",
        required=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="",
        type=str,
        default=None,
        required=False,
        help="The configuration file (contains all the information related to the training/inference session). "
        "Arg value is used if no config in model is found.",
    )

    args = parser.parse_args()
    _optimize_model(model=args.model, config=args.config, output_path=args.outputdir)


if __name__ == "__main__":
    old_way()
