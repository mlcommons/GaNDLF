#!usr/bin/env python
# -*- coding: utf-8 -*-

import os, argparse, sys
import click
from deprecated import deprecated

from GANDLF.entrypoints import append_copyright_to_help


def _verify_install():
    try:
        import GANDLF as gf

        print("GaNDLF installed version:", gf.__version__)
    except:
        raise Exception(
            "GaNDLF not properly installed, please see https://mlcommons.github.io/GaNDLF/setup"
        )

    # we always want to do submodule update to ensure any hash updates are ingested correctly
    try:
        os.system(f"{sys.executable} -m pip install -e .")
        # TODO: how do this works? why do we trigger on git? what if gandlf is installed as pypi package?
    except:
        print("Git was not found, please try again.")
    print("GaNDLF is ready. See https://mlcommons.github.io/GaNDLF/usage")


@click.command()
@append_copyright_to_help
def new_way():
    """ "Verify GaNDLF installation."""
    _verify_install()


# main function
@deprecated(
    "This is a deprecated way of running GanDLF. Please, use `gandlf verify-install` cli command "
    + "instead of `gandlf_verifyInstall`.\n"
    + "`gandlf_verifyInstall` script would be deprecated soon."
)
def old_way():
    parser = argparse.ArgumentParser(
        prog="GANDLF_VerifyInstall",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Verify GaNDLF installation.",
    )
    args = parser.parse_args()
    _verify_install()


if __name__ == "__main__":
    old_way()
