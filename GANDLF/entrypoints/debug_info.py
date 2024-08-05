#!usr/bin/env python
# -*- coding: utf-8 -*-
import platform
import argparse
from pip import main
from deprecated import deprecated
import click

from GANDLF import __version__
from GANDLF.entrypoints import append_copyright_to_help
from GANDLF.utils import get_git_hash
from GANDLF.utils import logger_setup

from GANDLF.cli import copyrightMessage


def _debug_info(verbose: bool):
    """Function to display necessary debugging information."""
    print(f"GANDLF version: {__version__}")
    print(f"Git hash: {get_git_hash()}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {(' ').join(list(platform.architecture()))}")
    print("Python environment:")
    print(f"  Version: {platform.python_version()}")
    print(f"  Implementation: {platform.python_implementation()}")
    print(f"  Compiler: {platform.python_compiler()}")
    print(f"  Build: {(' ').join(list(platform.python_build()))}")
    if verbose:
        print("  Installed packages:")
        print(main(["list"]))


@click.command()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="If passed, prints all packages installed as well",
)
@click.option(
    "--log-file",
    type=click.Path(),
    default=None,
    help="Output file which will contain the logs.",
)
@append_copyright_to_help
def new_way(verbose: bool, log_file):
    """Displays detailed info about system environment: library versions, settings, etc."""

    logger_setup(log_file)
    _debug_info(verbose=verbose)


# main function
@deprecated(
    "This is a deprecated way of running GanDLF. Please, use `gandlf debug-info` cli command "
    + "instead of `gandlf_debugInfo`.\n"
    + "`gandlf_debugInfo` script would be deprecated soon."
)
def old_way():
    parser = argparse.ArgumentParser(
        prog="GANDLF_DebugInfo",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate debugging information for maintainers.\n\n"
        + copyrightMessage,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="If True, prints all packages installed as well.",
    )
    args = parser.parse_args()
    logger_setup()
    _debug_info(verbose=args.verbose)


if __name__ == "__main__":
    old_way()
