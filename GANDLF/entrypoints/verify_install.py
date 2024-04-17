#!usr/bin/env python
# -*- coding: utf-8 -*-

import os, argparse, sys


# main function
def main():
    argparse.ArgumentParser(
        prog="GANDLF_VerifyInstall",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Verify GaNDLF installation.",
    )

    try:
        import GANDLF as gf

        print("GaNDLF installed version:", gf.__version__)
    except Exception as e:
        raise Exception(
            "GaNDLF not properly installed, please see https://mlcommons.github.io/GaNDLF/setup"
        ) from e

    # we always want to do submodule update to ensure any hash updates are ingested correctly
    try:
        os.system(f"{sys.executable} -m pip install -e .")
    except Exception as e:
        raise Exception("Git was not found, please try again.") from e

    print("GaNDLF is ready. See https://mlcommons.github.io/GaNDLF/usage")


if __name__ == "__main__":
    main()
