#!usr/bin/env python
# -*- coding: utf-8 -*-

import os, argparse, sys


# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GANDLF_VerifyInstall",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Verify GaNDLF installation.",
    )

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
    except:
        print("Git was not found, please try again.")

    args = parser.parse_args()

    print("GaNDLF is ready. See https://mlcommons.github.io/GaNDLF/usage")
