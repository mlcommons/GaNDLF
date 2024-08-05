#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast
import os
from typing import Optional

import click
from deprecated import deprecated
from GANDLF.cli import (
    deploy_targets,
    mlcube_types,
    run_deployment,
    recover_config,
    copyrightMessage,
)
from GANDLF.entrypoints import append_copyright_to_help
from GANDLF.utils import logger_setup


def _deploy(
    model: Optional[str],
    config: Optional[str],
    target: str,
    mlcube_type: str,
    mlcube_root: str,
    output_dir: str,
    requires_gpu: bool,
    entrypoint: Optional[str],
):
    os.makedirs(output_dir, exist_ok=True)

    default_config = os.path.join(output_dir, "original_config.yml")
    if not config and mlcube_type == "model":
        result = recover_config(model, default_config)
        assert (
            result
        ), "Error: No config was specified but automatic config extraction failed."
        config = default_config

    if not model and mlcube_type == "model":
        raise AssertionError(
            "Error: a path to a model directory should be provided when deploying a model"
        )
    print(f"{mlcube_root=}")
    print(f"{output_dir=}")
    print(f"{target=}")
    print(f"{mlcube_type=}")
    print(f"{entrypoint=}")
    print(f"{config=}")
    print(f"{model=}")
    print(f"{requires_gpu=}")

    result = run_deployment(
        mlcubedir=mlcube_root,
        outputdir=output_dir,
        target=target,
        mlcube_type=mlcube_type,
        entrypoint_script=entrypoint,
        configfile=config,
        modeldir=model,
        requires_gpu=requires_gpu,
    )

    assert result, "Deployment to the target platform failed."


@click.command()
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the model directory you wish to deploy. Required for model MLCubes, "
    "ignored for metrics MLCubes.",
)
@click.option(
    "--config",
    "-c",
    help="Optional path to an alternative config file to be embedded with the model. "
    "If blank/default, we use the previous config from the model instead. "
    "Only relevant for model MLCubes. Ignored for metrics MLCubes",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--target",
    "-t",
    required=True,
    type=click.Choice(deploy_targets),
    help="The target platform.",
)
@click.option(
    "--mlcube-type",
    type=click.Choice(mlcube_types),
    required=True,
    help="The mlcube type.",
)
@click.option(
    "--mlcube-root",
    "-r",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to an alternative MLCUBE_ROOT directory to use as a template. The source "
    "repository contains an example (https://github.com/mlcommons/GaNDLF/tree/master/mlcube).",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    help="Output directory path. "
    "For MLCube builds, generates an MLCube directory to be distributed with your MLCube.",
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option(
    "--requires-gpu/--no-gpu",
    "-g",
    is_flag=True,
    default=True,
    help="True if the model requires a GPU by default, False otherwise. "
    "Only relevant for model MLCubes. Ignored for metrics MLCubes",
)
@click.option(
    "--entrypoint",
    "-e",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="An optional custom python entrypoint script to use instead of the default specified in mlcube.yaml."
    " (Only for inference and metrics)",
)
@click.option(
    "--log-file",
    type=click.Path(),
    default=None,
    help="Output file which will contain the logs.",
)
@append_copyright_to_help
def new_way(
    model: Optional[str],
    config: Optional[str],
    target: str,
    mlcube_type: str,
    mlcube_root: str,
    output_dir: str,
    requires_gpu: bool,
    entrypoint: Optional[str],
    log_file: str,
):
    """Generate frozen/deployable versions of trained GaNDLF models."""
    logger_setup(log_file)
    _deploy(
        model=model,
        config=config,
        target=target,
        mlcube_type=mlcube_type,
        mlcube_root=mlcube_root,
        output_dir=output_dir,
        requires_gpu=requires_gpu,
        entrypoint=entrypoint,
    )


@deprecated(
    "This is a deprecated way of running GanDLF. Please, use `gandlf deploy` cli command "
    + "instead of `gandlf_deploy`. Note that in new CLI tool some params were renamed or changed its behavior:\n"
    + "  --outputdir to --output-dr\n"
    + "  --requires-gpu/-g now works as flag: True by default or if flag is passed. To disable gpu, use `--no-gpu` option\n"
    + "`gandlf_deploy` script would be deprecated soon."
)
def old_way():
    logger_setup()
    parser = argparse.ArgumentParser(
        prog="GANDLF_Deploy",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate frozen/deployable versions of trained GaNDLF models.\n\n"
        + copyrightMessage,
    )

    parser.add_argument(
        "-m",
        "--model",
        metavar="",
        type=str,
        help="Path to the model directory you wish to deploy. Required for model MLCubes, ignored for metrics MLCubes.",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="",
        type=str,
        default=None,
        help="Optional path to an alternative config file to be embedded with the model. If blank/default, we use the previous config from the model instead. Only relevant for model MLCubes. Ignored for metrics MLCubes",
    )
    parser.add_argument(
        "-t",
        "--target",
        metavar="",
        type=str,
        help="The target platform. Valid inputs are: "
        + ", ".join(deploy_targets)
        + " .",
        required=True,
    )
    parser.add_argument(
        "--mlcube-type",
        metavar="",
        type=str,
        help="The mlcube type. Valid inputs are: " + ", ".join(mlcube_types) + " .",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--mlcube-root",
        metavar="",
        type=str,
        required=True,
        help="Path to an alternative MLCUBE_ROOT directory to use as a template. The source repository contains an example (https://github.com/mlcommons/GaNDLF/tree/master/mlcube).",
    )
    parser.add_argument(
        "-o",
        "--outputdir",
        metavar="",
        type=str,
        help="Output directory path. For MLCube builds, generates an MLCube directory to be distributed with your MLCube.",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--requires-gpu",
        metavar="",
        type=ast.literal_eval,
        help="True if the model requires a GPU by default, False otherwise. Only relevant for model MLCubes. Ignored for metrics MLCubes",
        default=True,
    )
    parser.add_argument(
        "-e",
        "--entrypoint",
        metavar="",
        type=str,
        help="An optional custom python entrypoint script to use instead of the default specified in mlcube.yaml. (Only for inference and metrics)",
        default=None,
    )

    args = parser.parse_args()

    _deploy(
        model=args.model,
        config=args.config,
        target=args.target,
        mlcube_type=args.mlcube_type,
        mlcube_root=args.mlcube_root,
        output_dir=args.outputdir,
        requires_gpu=args.requires_gpu,
        entrypoint=args.entrypoint,
    )
