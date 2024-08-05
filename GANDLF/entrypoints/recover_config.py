#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from typing import Optional

import click
from deprecated import deprecated

from GANDLF.cli import copyrightMessage, recover_config
from GANDLF.entrypoints import append_copyright_to_help
from GANDLF.utils import logger_setup


def _recover_config(model_dir: Optional[str], mlcube: bool, output_file: str):
    if mlcube:
        search_dir = "/embedded_model/"
    else:
        search_dir = model_dir

    print(f"{model_dir=}")
    print(f"{mlcube=}")
    print(f"{search_dir=}")
    print(f"{output_file=}")
    result = recover_config(search_dir, output_file)
    assert result, "Config file recovery failed."


@click.command()
@click.option(
    "--model-dir",
    "-m",
    help="Path to the model directory.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--mlcube",
    "-c",
    is_flag=True,
    help="Pass this option to attempt to extract the config from the embedded model in a GaNDLF MLCube "
    "(if any). Only useful in that context. If passed, model-dir param is ignored.",
)
@click.option(
    "--output-file",
    "-o",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False),
    help="Path to an output file where the config will be written.",
)
@click.option(
    "--log-file",
    type=click.Path(),
    default=None,
    help="Output file which will contain the logs.",
)
@append_copyright_to_help
def new_way(model_dir, mlcube, output_file, log_file):
    """Recovers a config file from a GaNDLF model. If used from within a deployed GaNDLF MLCube,
    attempts to extract the config from the embedded model."""

    logger_setup(log_file)
    _recover_config(model_dir=model_dir, mlcube=mlcube, output_file=output_file)


# old-fashioned way of running gandlf via `gandlf_recoverConfig`.
@deprecated(
    "This is a deprecated way of running GanDLF. Please, use `gandlf recover-config` cli command "
    + "instead of `gandlf_recoverConfig`. Note that in new CLI tool some params were renamed or changed its behavior:\n"
    + "  --modeldir to --model-dir\n"
    + "  --mlcube is flag now with default False if not passed. Does not require to pass any additional values\n"
    + "  --outputFile to --output-file`\n"
    + "`gandlf_recoverConfig` script would be deprecated soon."
)
def old_way():
    logger_setup()
    parser = argparse.ArgumentParser(
        prog="GANDLF_RecoverConfig",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Recovers a config file from a GaNDLF model. If used from within a deployed GaNDLF MLCube, attempts to extract the config from the embedded model.\n\n"
        + copyrightMessage,
    )

    parser.add_argument(
        "-m", "--modeldir", metavar="", type=str, help="Path to the model directory."
    )
    # TODO: despite of `str` type, real value is never used (only checks if it is filled or not)
    #  Thus, caveats:
    #    * passing `--mlcube False` would still process it as mlcube;
    #    * passing `--mlcube "" ` (with empty str) acts as non-mlcube
    parser.add_argument(
        "-c",
        "--mlcube",
        metavar="",
        type=str,
        help="Pass this option to attempt to extract the config from the embedded model in a GaNDLF MLCube (if any). Only useful in that context. If passed, model-dir param is ignored.",
    )
    parser.add_argument(
        "-o",
        "--outputFile",
        metavar="",
        type=str,
        required=True,
        help="Path to an output file where the config will be written.",
    )

    args = parser.parse_args()

    _recover_config(args.modeldir, args.mlcube, args.outputFile)


if __name__ == "__main__":
    old_way()
