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
    "numpy==1.22.0",
    "scipy",
    "SimpleITK!=2.0.*",
    "torchvision",
    "tqdm",
    "torchio==0.18.75",
    "pandas",
    "scikit-learn>=0.23.2",
    "scikit-image>=0.19.1",
    'pickle5>=0.0.11; python_version < "3.8.0"',
    "setuptools",
    "seaborn",
    "pyyaml",
    "tiffslide",
    "matplotlib",
    "requests>=2.25.0",
    "pytest",
    "coverage",
    "pytest-cov",
    "psutil",
    "medcam",
    "opencv-python",
    "torchmetrics==0.5.1",  # newer versions have changed api for f1 invocation
    "OpenPatchMiner==0.1.8",
    "zarr==2.10.3",
    "pydicom",
    "onnx",
    "torchinfo==1.7.0",
    "segmentation-models-pytorch==0.3.0",
    "ACSConv==0.1.1",
]

# pytorch doesn't have LTS support on OSX - https://github.com/CBICA/GaNDLF/issues/389
if sys.platform == "darwin":
    requirements.append("torch==1.11.0")
else:
    requirements.append("torch==1.11.0")

setup(
    name="GANDLF",
    version=__version__,
    author="Jose Agraz, Vinayak Ahluwalia, Bhakti Baheti, Spyridon Bakas, Ujjwal Baid, Megh Bhalerao, Brandon Edwards, Karol Gotkowski, Caleb Grenko, Orhun Güley, Ibrahim Ethem Hamamci, Sarthak Pati, Micah Sheller, Juliia Skobleva, Siddhesh Thakur, Spiros Thermos",  # alphabetical order
    author_email="software@cbica.upenn.edu",
    python_requires=">=3.7",
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
        "gandlf_verifyInstall",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
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
