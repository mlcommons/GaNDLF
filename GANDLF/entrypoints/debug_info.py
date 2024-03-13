#!usr/bin/env python
# -*- coding: utf-8 -*-
import platform
from deprecated import deprecated
import click

from GANDLF import __version__
from GANDLF.entrypoints import append_copyright_to_help
from GANDLF.utils import get_git_hash


def _debug_info():
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


@click.command()
@append_copyright_to_help
def new_way():
    """Displays detailed info about system environment: library versions, settings, etc."""
    _debug_info()


# main function
@deprecated("This is a deprecated way of running GanDLF. Please, use `gandlf debug-info` cli command " +
            "instead of `gandlf_debugInfo`.\n" +
            "`gandlf_debugInfo` script would be deprecated soon.")
def old_way():
    _debug_info()


if __name__ == '__main__':
    old_way()
