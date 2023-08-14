#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast
import os
from GANDLF.cli import (
    deploy_targets,
    mlcube_types,
    run_deployment,
    recover_config,
    copyrightMessage,
)


if __name__ == "__main__":
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
        help="Path to an alternative MLCUBE_ROOT directory to use as a template (or a path to a specific mlcube YAML configuration file, in which case we will use the parent directory). The source repository contains an example (https://github.com/mlcommons/GaNDLF/tree/master/mlcube).",
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

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir, exist_ok=True)

    config_to_use = args.config
    if not args.config and args.mlcube_type == "model":
        result = recover_config(args.model, args.outputdir + "/original_config.yml")
        assert (
            result
        ), "Error: No config was specified but automatic config extraction failed."
        config_to_use = args.outputdir + "/original_config.yml"

    if not args.model and args.mlcube_type == "model":
        raise AssertionError(
            "Error: a path to a model directory should be provided when deploying a model"
        )
    result = run_deployment(
        args.mlcube_root,
        args.outputdir,
        args.target,
        args.mlcube_type,
        args.entrypoint,
        config_to_use,
        args.model,
        args.requires_gpu,
    )
    assert result, "Deployment to the target platform failed."
