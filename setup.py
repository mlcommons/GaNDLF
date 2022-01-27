#!/usr/bin/env python

"""The setup script."""


import os
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

with open("README.md") as readme_file:
    readme = readme_file.read()


def git_submodule_update():
    ## submodule update
    os.system("git submodule update --init --recursive")


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        git_submodule_update()


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        git_submodule_update()


class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        git_submodule_update()


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
    "black",
    "numpy==1.21.0",
    "scipy",
    "SimpleITK==2.1.0",
    "torch>=1.7",
    "torchvision",
    "tqdm",
    "torchio==0.18.57",
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
    "pytest-cov",
    "psutil",
    "medcam",
    "opencv-python",
    "torchmetrics",
    "OpenPatchMiner==0.1.6",
    "pydicom",
]

setup(
    name="GANDLF",
    version=__version__,
    author="Jose Agraz, Vinayak Ahluwalia, Bhakti Baheti, Spyridon Bakas, Ujjwal Baid, Megh Bhalerao, Brandon Edwards, Karol Gotkowski, Caleb Grenko, Orhun Güley, Ibrahim Ethem Hamamci, Sarthak Pati, Micah Sheller, Juliia Skobleva, Siddhesh Thakur, Spiros Thermos",  # alphabetical order
    author_email="software@cbica.upenn.edu",
    python_requires=">=3.6",
    packages=find_packages(),
    cmdclass={  # this ensures git_submodule_update is called during install
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
        "egg_info": CustomEggInfoCommand,
    },
    scripts=[
        "gandlf_run",
        "gandlf_constructCSV",
        "gandlf_collectStats",
        "gandlf_patchMiner",
        "gandlf_preprocess",
        "gandlf_anonymizer",
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
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
    ],
    description=(
        "PyTorch-based framework that handles segmentation/regression/classification using various DL architectures for medical imaging."
    ),
    install_requires=requirements,
    license="BSD-3-Clause License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="semantic, segmentation, regression, classification, data-augmentation, medical-imaging",
    zip_safe=False,
)

## windows vips installation
if os.name == "nt":  # proceed for windows
    from pathlib import Path

    # download and extract if main dll is absent
    if not Path("./vips/vips-dev-8.10/bin/libvips-42.dll").exists():
        print("Downloading and extracting VIPS for Windows")
        url = "https://github.com/libvips/libvips/releases/download/v8.10.2/vips-dev-w64-all-8.10.2.zip"
        zip_to_extract = "./vips.zip"
        import urllib.request, zipfile

        urllib.request.urlretrieve(url, zip_to_extract)
        z = zipfile.ZipFile(zip_to_extract)
        z.extractall("./vips")
        z.close()
        os.remove(zip_to_extract)
