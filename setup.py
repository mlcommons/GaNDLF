#!/usr/bin/env python

"""The setup script."""


import sys, re
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

try:
    with open("README.md") as readme_file:
        readme = readme_file.read()
except Exception as error:
    readme = "No README information found."
    sys.stderr.write("Warning: Could not open '%s' due %s\n" % ("README.md", error))


class CustomInstallCommand(install):
    def run(self):
        install.run(self)


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)


class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)


try:
    filepath = "GANDLF/version.py"
    version_file = open(filepath)
    (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())

except Exception as error:
    __version__ = "0.0.1"
    sys.stderr.write("Warning: Could not open '%s' due %s\n" % (filepath, error))

requirements = [
    "torch==1.13.1",
    "black",
    "numpy==1.22.0",
    "scipy",
    "SimpleITK!=2.0.*",
    "SimpleITK!=2.2.1",  # https://github.com/mlcommons/GaNDLF/issues/536
    "torchvision",
    "tqdm",
    "torchio==0.18.75",
    "pandas",
    "scikit-learn>=0.23.2",
    "scikit-image>=0.19.1",
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
    "docker",
    "dicom-anonymizer",
    "twine",
    "zarr",
    "keyring",
]

if __name__ == "__main__":
    setup(
        name="GANDLF",
        version=__version__,
        author="MLCommons",
        author_email="gandlf@mlcommons.org",
        python_requires=">=3.8",
        packages=find_packages(),
        cmdclass={
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
            "gandlf_configGenerator",
            "gandlf_recoverConfig",
            "gandlf_deploy",
            "gandlf_optimizeModel",
        ],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Medical Science Apps.",
        ],
        description=(
            "PyTorch-based framework that handles segmentation/regression/classification using various DL architectures for medical imaging."
        ),
        install_requires=requirements,
        license="Apache-2.0",
        long_description=readme,
        long_description_content_type="text/markdown",
        include_package_data=True,
        keywords="semantic, segmentation, regression, classification, data-augmentation, medical-imaging, clinical-workflows, deep-learning, pytorch",
        zip_safe=False,
    )
