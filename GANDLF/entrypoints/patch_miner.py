#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
from typing import Optional
from deprecated import deprecated
import click

from GANDLF.cli.patch_extraction import patch_extraction
from GANDLF.cli import copyrightMessage
from GANDLF.entrypoints import append_copyright_to_help


def _mine_patches(input_path: str, output_path: str, config: Optional[str]):
    patch_extraction(input_path, output_path, config)
    logging.info("Finished.")


@click.command()
@click.option(
    "--input-csv",
    "-i",  # TODO: check - really csv only fits?
    # TODO: should we rename it to --input-path?
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="input path for the tissue",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="output path for the patches",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
    help="config (in YAML) for running the patch miner. Needs 'scale' and 'patch_size' to be defined, "
    "otherwise defaults to 16 and (256, 256), respectively.",
)
@append_copyright_to_help
def new_way(input_csv: str, output_path: str, config: Optional[str]):
    """Construct patches from whole slide image(s)."""
    _mine_patches(input_path=input_csv, output_path=output_path, config=config)


@deprecated(
    "This is a deprecated way of running GanDLF. Please, use `gandlf patch-miner` cli command "
    + "instead of `gandlf_patchMiner`. Note that in new CLI tool some params were renamed to snake-case:\n"
    + "  --input_CSV to --input-csv\n"
    + "  --output_path to --output-path\n"
    + "`gandlf_patchMiner` script would be deprecated soon."
)
def old_way():
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
    _mine_patches(
        input_path=args.input_path, output_path=args.output_path, config=args.config
    )


if __name__ == "__main__":
    old_way()
