# Extending GaNDLF

For any new feature, please ensure the corresponding option in the [sample configuration](https://github.com/CBICA/GaNDLF/blob/master/samples/sample_training.yaml) is added, so that others can review/use/extend it as needed.

## Table of Contents
- [Extending GaNDLF](#extending-gandlf)
  - [Table of Contents](#table-of-contents)
  - [Environment](#environment)
  - [Architecture](#architecture)
  - [Adding Models](#adding-models)
  - [Adding Augmentation Transformations](#adding-augmentation-transformations)
  - [Adding Preprocessing functionality](#adding-preprocessing-functionality)
  - [Adding Training Functionality](#adding-training-functionality)
  - [Adding Inference Functionality](#adding-inference-functionality)
  - [Update Tests](#update-tests)
  - [Run Tests](#run-tests)
    - [Code coverage](#code-coverage)

## Environment

Before starting to work on the code-level on GaNDLF, please get the environment ready:

**NOTE**: Windows users, please ensure sure you have [Microsoft Visual C++ 14.0 or greater](http://visualstudio.microsoft.com/visual-cpp-build-tools) installed.

```bash
git clone https://github.com/CBICA/GaNDLF.git
cd GaNDLF
conda create -p ./venv python=3.6 -y
conda activate ./venv
conda install -c conda-forge mamba -y # allows for faster dependency solving
mamba install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge gandlf -y
pip install -e .
```

[Back To Top &uarr;](#table-of-contents)

## Architecture

- Command-line parsing: [gandlf_run](https://github.com/CBICA/GaNDLF/blob/master/gandlf_run)
- Parameters from [training configuration](https://github.com/CBICA/GaNDLF/blob/master/samples/config_all_options.yaml) get passed as a `dict` via [parameter parser](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/parseConfig.py)
- [Training Manager](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/training_manager.py): 
  - Handles k-fold training 
  - Main entry point from CLI
- [Training Function](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/compute/training_loop.py): 
  - Performs actual training
- [Inference Manager](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/inference_manager.py): 
  - Handles inference functionality 
  - Main entry point from CLI
- [Inference Function](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/compute/inference_loop.py): 
  - Performs actual inference

[Back To Top &uarr;](#table-of-contents)

## Adding Models

- For details, please see [README for `GANDLF.models` submodule](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/models/Readme.md).
- [Update Tests](#update-tests)

[Back To Top &uarr;](#table-of-contents)

## Adding Augmentation Transformations

- Update or add dependency in [setup](https://github.com/CBICA/GaNDLF/blob/master/setup.py), if appropriate.
- Add transformation to `global_augs_dict`, defined in [`GANDLF/data/augmentation/__init__.py`](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/data/augmentation/__init__.py)
- Ensure probability is used as input; probability is not used for any [preprocessing operations](https://github.com/CBICA/GaNDLF/tree/master/GANDLF/data/preprocessing)
- For details, please see [README for `GANDLF.data.augmentation` submodule](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/data/augmentation/README.md).
- [Update Tests](#update-tests)

[Back To Top &uarr;](#table-of-contents)

## Adding Preprocessing functionality

- Update or add dependency in [setup](https://github.com/CBICA/GaNDLF/blob/master/setup.py), if appropriate.
- All transforms should be defined by inheriting from `torchio.transforms.intensity_transform.IntensityTransform`. For example, please see the threshold/clip functionality in the [`GANDLF/data/preprocessing/threshold_and_clip.py`](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/data/preprocessing/threshold_and_clip.py) file.
- Define each option in the configuration file under the correct key (again, see threshold/clip as examples)
- Add transformation to `global_preprocessing_dict`, defined in [`GANDLF/data/preprocessing/__init__.py`](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/data/preprocessing/__init__.py)
- For details, please see [README for `GANDLF.data.preprocessing` submodule](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/data/preprocessing/README.md).
- [Update Tests](#update-tests)

[Back To Top &uarr;](#table-of-contents)

## Adding Training Functionality

- Update [Training Function](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/compute/training_loop.py)
- Update [Training Manager](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/training_manager.py), if any training API has changed
- [Update Tests](#update-tests)

[Back To Top &uarr;](#table-of-contents)

## Adding Inference Functionality

- Update [Inference Function](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/compute/inference_loop.py)
- Update [Inference Manager](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/inference_manager.py), if any inference API has changed
- [Update Tests](#update-tests)

[Back To Top &uarr;](#table-of-contents)

## Update Tests

Once you have made changes to functionality, it is imperative that the unit tests be updated to cover the new code. Please see the [full testing suite](https://github.com/CBICA/GaNDLF/blob/master/testing/test_full.py) for details and examples.

[Back To Top &uarr;](#table-of-contents)

## Run Tests

Once you have the virtual environment set up, tests can be run using the following command:
```powershell
pytest --device cuda # can be cuda or cpu, defaults to cpu
```

[Back To Top &uarr;](#table-of-contents)

### Code coverage

The code coverage for the tests can be obtained by the following command:
```powershell
coverage run -m pytest --device cuda; coverage report -m
```

[Back To Top &uarr;](#table-of-contents)
