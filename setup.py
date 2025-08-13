#!/usr/bin/env python

"""The setup script."""


import sys, re, os
from setuptools import setup, find_packages


try:
    with open("README.md") as readme_file:
        readme = readme_file.read()
except Exception as error:
    readme = "No README information found."
    sys.stderr.write("Warning: Could not open '%s' due %s\n" % ("README.md", error))

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

# Any extra files should be located at `GANDLF` module folder (not in repo root)
extra_files_root = ["logging_config.yaml"]
extra_files_metrics = ["panoptica_config_brats.yaml"]
toplevel_package_excludes = ["testing*"]

# specifying version for `black` separately because it is also used to [check for lint](https://github.com/mlcommons/GaNDLF/blob/master/.github/workflows/black.yml)
black_version = "23.11.0"
requirements = [
    "torch==2.7.1",
    f"black=={black_version}",
    "lightning==2.5.2",
    "numpy==1.26.4",
    "scipy",
    "SimpleITK!=2.0.*",
    "SimpleITK!=2.2.1",  # https://github.com/mlcommons/GaNDLF/issues/536
    "torchvision",
    "tqdm",
    "torchio==0.19.6",
    "pandas>=2.0.0",
    "scikit-learn>=0.23.2",
    "scikit-image>=0.19.1",
    "setuptools",
    "seaborn",
    "pyyaml",
    "matplotlib",
    "gdown==5.1.0",
    "overrides==7.7.0",
    "pytest",
    "coverage",
    "pytest-cov",
    "psutil",
    "medcam",
    "opencv-python",
    "torchmetrics==1.1.2",
    "zarr==2.10.3",
    "numcodecs<0.16.0",
    "pydicom",
    "onnx",
    "torchinfo==1.7.0",
    "segmentation-models-pytorch==0.4.0",
    "ACSConv==0.1.1",
    # https://github.com/docker/docker-py/issues/3256
    "requests>=2.32.2",
    "docker",
    "dicom-anonymizer==1.0.12",
    "twine",
    "keyring",
    "monai==1.4.0",
    "click==8.1.8",
    "deprecated",
    "packaging==24.0",
    "colorlog",
    "opacus==1.5.2",
    "huggingface-hub==0.25.1",
    "openslide-bin",
    "openslide-python==1.4.1",
    "lion-pytorch==0.2.2",
    "pydantic==2.10.6",
    "panoptica>=1.4.1",
]

if __name__ == "__main__":
    setup(
        name="GANDLF",
        version=__version__,
        author="MLCommons",
        author_email="gandlf@mlcommons.org",
        python_requires=">3.9, <3.13",
        packages=find_packages(
            where=os.path.dirname(os.path.abspath(__file__)),
            exclude=toplevel_package_excludes,
        ),
        entry_points={
            "console_scripts": [
                "gandlf=GANDLF.entrypoints.cli_tool:gandlf",
                # old entrypoints
                "gandlf_run=GANDLF.entrypoints.run:old_way",
                "gandlf_constructCSV=GANDLF.entrypoints.construct_csv:old_way",
                "gandlf_collectStats=GANDLF.entrypoints.collect_stats:old_way",
                "gandlf_patchMiner=GANDLF.entrypoints.patch_miner:old_way",
                "gandlf_preprocess=GANDLF.entrypoints.preprocess:old_way",
                "gandlf_anonymizer=GANDLF.entrypoints.anonymizer:old_way",
                "gandlf_configGenerator=GANDLF.entrypoints.config_generator:old_way",
                "gandlf_verifyInstall=GANDLF.entrypoints.verify_install:old_way",
                "gandlf_recoverConfig=GANDLF.entrypoints.recover_config:old_way",
                "gandlf_deploy=GANDLF.entrypoints.deploy:old_way",
                "gandlf_optimizeModel=GANDLF.entrypoints.optimize_model:old_way",
                "gandlf_generateMetrics=GANDLF.entrypoints.generate_metrics:old_way",
                "gandlf_debugInfo=GANDLF.entrypoints.debug_info:old_way",
                "gandlf_splitCSV=GANDLF.entrypoints.split_csv:old_way",
            ]
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
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
        package_data={
            "GANDLF": extra_files_root,
            "GANDLF.metrics": extra_files_metrics,
        },
        keywords="semantic, segmentation, regression, classification, data-augmentation, medical-imaging, clinical-workflows, deep-learning, pytorch",
        zip_safe=False,
    )
