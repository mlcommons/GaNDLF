#!/usr/bin/env python

"""The setup script."""


import sys, re, os
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

# Handle cases where specific files need to be bundled into the final package as installed via PyPI
dockerfiles = [
    item
    for item in os.listdir(os.path.dirname(os.path.abspath(__file__)))
    if (os.path.isfile(item) and item.startswith("Dockerfile-"))
]
entrypoint_files = [
    item
    for item in os.listdir(os.path.dirname(os.path.abspath(__file__)))
    if (os.path.isfile(item) and item.startswith("gandlf_"))
]
setup_files = ["setup.py", ".dockerignore", "pyproject.toml", "MANIFEST.in"]
all_extra_files = dockerfiles + entrypoint_files + setup_files
all_extra_files_pathcorrected = [os.path.join("../", item) for item in all_extra_files]
# find_packages should only ever find these as subpackages of gandlf, not as top-level packages
# generate this dynamically?
# GANDLF.GANDLF is needed to prevent recursion madness in deployments
toplevel_package_excludes = [
    "GANDLF.GANDLF",
    "anonymize",
    "cli",
    "compute",
    "data",
    "grad_clipping",
    "losses",
    "metrics",
    "models",
    "optimizers",
    "schedulers",
    "utils",
]


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
    "pandas<2.0.0",
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
    "torchmetrics==0.8.1",
    "zarr==2.10.3",
    "pydicom",
    "onnx",
    "torchinfo==1.7.0",
    "segmentation-models-pytorch==0.3.2",
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
        packages=find_packages(
            where=os.path.dirname(os.path.abspath(__file__)),
            exclude=toplevel_package_excludes,
        ),
        cmdclass={
            "install": CustomInstallCommand,
            "develop": CustomDevelopCommand,
            "egg_info": CustomEggInfoCommand,
        },
        entry_points={
            "console_scripts": [
                "gandlf_anonymizer=GANDLF.cli.anonymize:main",
                "gandlf_constructCSV=GANDLF.cli.construct_csv:main",
                "gandlf_collectStats=GANDLF.cli.collect_stats:main",
                "gandlf_configGenerator=GANDLF.cli.config_generator:main",
                "gandlf_deploy=GANDLF.cli.deploy:main",
                "gandlf_generateMetrics=GANDLF.cli.generate_metrics:main",
                "gandlf_optimizeModel=GANDLF.cli.post_training_model_optimization:main",
                "gandlf_patchMiner=GANDLF.cli.patch_extraction:main",
                "gandlf_preprocess=GANDLF.cli.preprocess_and_save:main",
                "gandlf_recoverConfig=GANDLF.cli.recover_config:main",
                "gandlf_run=GANDLF.cli.main_run:main",
                "gandlf_verifyInstall=GANDLF.gandlf_verifyInstall:main",
            ]
        },
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
        package_data={"GANDLF": all_extra_files_pathcorrected},
        keywords="semantic, segmentation, regression, classification, data-augmentation, medical-imaging, clinical-workflows, deep-learning, pytorch",
        zip_safe=False,
    )
