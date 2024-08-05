#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast
import logging

import click
from deprecated import deprecated
from GANDLF.cli import preprocess_and_save, copyrightMessage
from GANDLF.entrypoints import append_copyright_to_help
from GANDLF.utils import logger_setup


def _preprocess(
    config: str,
    input_data: str,
    output_dir: str,
    label_pad: str,
    apply_augs: bool,
    crop_zero: bool,
):
    print(f"{config=}")
    print(f"{input_data=}")
    print(f"{output_dir=}")
    print(f"{label_pad=}")
    print(f"{apply_augs=}")
    print(f"{crop_zero=}")
    preprocess_and_save(
        data_csv=input_data,
        config_file=config,
        output_dir=output_dir,
        label_pad_mode=label_pad,
        applyaugs=apply_augs,
        apply_zero_crop=crop_zero,
    )

    # TODO: in `old_way` default logging level is warning, thus those 'finished' are not printed anymore
    logging.info("Finished.")


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="The configuration file (contains all the information related to the training/inference session),"
    " this is read from 'output' during inference",
)
@click.option(
    "--input-data",
    "-i",  # TODO: mention pickled df also fits
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Data csv file that is used for training/inference",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Output directory to save intermediate files and model weights",
)
@click.option(
    "--label-pad",
    "-l",
    type=str,
    default="constant",
    help="Specifies the padding strategy for the label when 'patch_sampler' is 'label'. "
    "Defaults to 'constant' [full list: https://numpy.org/doc/stable/reference/generated/numpy.pad.html]",
)
@click.option(
    "--apply-augs",
    "-a",
    is_flag=True,
    help="If passed, applies data augmentations during output creation",
)
@click.option(
    "--crop-zero",
    "-z",
    is_flag=True,
    help="If passed, applies zero cropping during output creation.",
)
@click.option(
    "--log-file",
    type=click.Path(),
    default=None,
    help="Output file which will contain the logs.",
)
@append_copyright_to_help
def new_way(
    config: str,
    input_data: str,
    output_dir: str,
    label_pad: str,
    apply_augs: bool,
    crop_zero: bool,
    log_file: str,
):
    """Generate training/inference data which are preprocessed to reduce resource footprint during computation."""

    logger_setup(log_file)
    _preprocess(
        config=config,
        input_data=input_data,
        output_dir=output_dir,
        label_pad=label_pad,
        apply_augs=apply_augs,
        crop_zero=crop_zero,
    )


@deprecated(
    "This is a deprecated way of running GanDLF. Please, use `gandlf preprocess` cli command "
    + "instead of `gandlf_preprocess`. Note that in new CLI tool some params were renamed to snake-case:\n"
    + "  --inputdata to --input-data\n"
    + "  --labelPad to --label-pad\n"
    + "  --applyaugs to --apply-augs; it is flag now, i.e. no value accepted\n"
    + "  --cropzero to --crop-zero;  it is flag now, i.e. no value accepted\n"
    + "`gandlf_preprocess` script would be deprecated soon."
)
def old_way():
    logger_setup()
    parser = argparse.ArgumentParser(
        prog="GANDLF_Preprocess",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate training/inference data which are preprocessed to reduce resource footprint during computation.\n\n"
        + copyrightMessage,
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="",
        type=str,
        help="The configuration file (contains all the information related to the training/inference session), this is read from 'output' during inference",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--inputdata",
        metavar="",
        type=str,
        help="Data csv file that is used for training/inference",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="",
        type=str,
        help="Output directory to save intermediate files and model weights",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--labelPad",
        metavar="",
        type=str,
        default="constant",
        help="This specifies the padding strategy for the label when 'patch_sampler' is 'label'. Defaults to 'constant' [full list: https://numpy.org/doc/stable/reference/generated/numpy.pad.html]",
        required=False,
    )
    parser.add_argument(
        "-a",
        "--applyaugs",
        metavar="",
        type=ast.literal_eval,
        default=False,
        help="This specifies the whether to apply data augmentation during output creation. Defaults to False",
        required=False,
    )
    parser.add_argument(
        "-z",
        "--cropzero",
        metavar="",
        type=ast.literal_eval,
        default=False,
        help="This specifies the whether to apply zero cropping during output creation. Defaults to False",
        required=False,
    )

    args = parser.parse_args()
    _preprocess(
        config=args.config,
        input_data=args.inputdata,
        output_dir=args.output,
        label_pad=args.labelPad,
        apply_augs=args.applyaugs,
        crop_zero=args.cropzero,
    )
