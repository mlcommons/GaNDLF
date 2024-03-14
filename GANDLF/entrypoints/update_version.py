#!usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import fileinput
from deprecated import deprecated
import click

from GANDLF.entrypoints import append_copyright_to_help


def _update_version(old_version: str, new_version: str):
    def in_place_string_replace(
        filename: str, old_string: str, new_string: str
    ) -> None:
        """
        Replace a string in a file in place.

        Args:
            filename (str): The file to replace the string in
            old_string (str): The string to replace
            new_string (str): The string to replace with
        """
        if os.path.exists(filename):
            with fileinput.FileInput(filename, inplace=True) as file:
                for line in file:
                    print(line.replace(old_string, new_string), end="")

    cwd = os.getcwd()
    in_place_string_replace(
        os.path.join(cwd, "../version.py"),
        old_version,
        new_version,
    )

    # find all yaml files in samples and testing directories
    folders_to_iterate = [
        os.path.join(cwd, "../../samples"),
        os.path.join(cwd, "../../testing"),
    ]

    files_where_version_is_stored = [
        os.path.join(cwd, "../../mlcube/model_mlcube/workspace/config.yml"),
        os.path.join(
            cwd, "../../tutorials/classification_medmnist_notebook/config.yaml"
        ),
    ]

    for folder in folders_to_iterate:
        if os.path.isdir(folder):
            files_in_dir = os.listdir(folder)
            for file in files_in_dir:
                if file.endswith(".yaml") or file.endswith(".yml"):
                    files_where_version_is_stored.append(os.path.join(folder, file))

    old_version = old_version.replace("-dev", "")
    new_version = new_version.replace("-dev", "")

    # update the version.py file
    for filename in files_where_version_is_stored:
        in_place_string_replace(filename, old_version, new_version)

    print("Version updated successfully in `version.py` and all configuration files!")


@click.command()
@click.option(
    "--old-version",
    "-ov",  # TODO: renamed from --old_version
    required=True,
    help="The old version number",
)
@click.option(
    "--new-version",
    "-nv",  # TODO: renamed from --new_version
    required=True,
    help="The new version number",
)
@append_copyright_to_help
def new_way(old_version: str, new_version: str):
    """Update versions when creating a new release of GaNDLF, also useful when updating the version for development."""
    _update_version(old_version=old_version, new_version=new_version)


@deprecated(
    "This is a deprecated way of running GanDLF. Please, use `gandlf update-version` cli command "
    + "instead of `gandlf_updateVersion`. Note that in new CLI tool some params were renamed:\n"
    + "  --old_version to --old-version\n"
    + "  --new_version to --new-version\n"
    + "`gandlf_updateVersion` script would be deprecated soon."
)
def old_way():
    parser = argparse.ArgumentParser(
        prog="GANDLF_UpdateVersion",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Update versions when creating a new release of GaNDLF, also useful when updating the version for development.\n\n",
    )
    parser.add_argument(
        "-ov",
        "--old_version",
        metavar="",
        type=str,
        required=True,
        help="The old version number",
    )
    parser.add_argument(
        "-nv",
        "--new_version",
        metavar="",
        type=str,
        required=True,
        help="The new version number",
    )

    args = parser.parse_args()

    _update_version(old_version=args.old_version, new_version=args.new_version)
