#!usr/bin/env python
# -*- coding: utf-8 -*-
import platform

from GANDLF import __version__
from GANDLF.utils import get_git_hash


if __name__ == "__main__":
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
