#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

# read version.py
import sys, re

try:
    filepath = "GANDLF/version.py"
    version_file = open(filepath)
    (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())

except Exception as error:
    __version__ = "0.0.1"
    sys.stderr.write("Warning: Could not open '%s' due %s\n" % (filepath, error))

requirements = [
    "numpy==1.19.2",
    "scipy",
    "SimpleITK==1.2.4",
    "torch>=1.7",
    "torchvision",
    "tqdm",
    "torchio==0.18.29",
    "pandas",
    "pylint",
    "scikit-learn==0.23.1",
    "pickle5==0.0.11",
    "setuptools",
    "seaborn",
    "pyyaml",
    "openslide-python",
    "scikit-image",
    "matplotlib",
    "requests>=2.25.0",
    "pyvips",
    "pytest",
    "coverage",
    "psutil",
    "medcam",
    "torchmetrics",
]

setup(
    name="GANDLF",
    version=__version__,
    author="Jose Agraz, Ujjwal Baid, Megh Bhalerao, Brandon Edwards, Karol Gotkowski, Caleb Grenko, Sarthak Pati, Micah Sheller, Siddhesh Thakur",  # alphabetical order
    author_email="software@cbica.upenn.edu",
    python_requires=">=3.6",
    packages=find_packages(),
    scripts=[
        "gandlf_run",
        "gandlf_constructCSV",
        "gandlf_collectStats",
        "gandlf_patchMiner",
        "gandlf_preprocess",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description=(
        "PyTorch-based framework that handles segmentation/regression/classification using various DL architectures for medical imaging."
    ),
    install_requires=requirements,
    license="BSD-3-Clause License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="semantic, segmentation, regression, classification, brain, breast, liver, lung, augmentation, medical-imaging",
    zip_safe=False,
)

import os

## submodule update
os.system("git submodule update --init --recursive")

## windows vips installation
if os.name == "nt":  # proceed for windows
    from pathlib import Path

    if not Path(
        "./vips/vips-dev-8.10/bin/libvips-42.dll"
    ).exists():  # download and extract if main dll is absent
        print("Downloading and extracting VIPS for Windows")
        url = "https://github.com/libvips/libvips/releases/download/v8.10.2/vips-dev-w64-all-8.10.2.zip"
        zip_to_extract = "./vips.zip"
        import urllib.request, zipfile

        urllib.request.urlretrieve(url, zip_to_extract)
        z = zipfile.ZipFile(zip_to_extract)
        z.extractall("./vips")
        z.close()
        os.remove(zip_to_extract)
