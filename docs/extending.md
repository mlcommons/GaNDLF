# Extending GaNDLF

For any new feature, please ensure the corresponding option in the [sample configuration](https://github.com/CBICA/GaNDLF/blob/master/samples/sample_training.yaml) is added, so that others can review/use/extend it as needed.

## Table of Contents
- [Environment](#environment)
- [Architecture](#architecture)
- [Adding Models](#adding-models)
- [Adding Transformations](#adding-transformations)
- [Adding Pre-processing functionality](#adding-pre-processing-functionality)
- [Adding Training functionality](#adding-training-functionality)
- [Adding Inference functionality](#adding-inference-functionality)
- [Update Tests](#update-tests)
- [Run Tests](#run-tests)
- [Code Coverage](#code-coverage)

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
## For windows
# mamba install -c sdvillal openslide -y
## For linux
# mamba install -c conda-forge libvips openslide -y
conda install -c conda-forge gandlf -y
pip install -e .
```

[Back To Top &uarr;](#table-of-contents)

## Architecture

- Command-line parsing: [gandlf_run](https://github.com/CBICA/GaNDLF/blob/master/gandlf_run)
- Parameters from [training configuration](https://github.com/CBICA/GaNDLF/blob/master/samples/sample_training.yaml) getting passed as dict via [parameter parser](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/parseConfig.py)
- [Training Manager](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/training_manager.py): 
  - Handles k-fold training 
  - Main entry point from CLI
- [Training Function](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/training_loop.py): 
  - Performs actual training
- [Inference Manager](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/inference_manager.py): 
  - Handles inference functionality 
  - Main entry point from CLI
- [Inference Function](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/inference_loop.py): 
  - Performs actual inference

[Back To Top &uarr;](#table-of-contents)

## Adding Models

- Add model code in [./GANDLF/models/](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/models/)
- Update initialization in [./GANDLF/parameterParsing](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/parameterParsing.py)
- Ensure both 2D and 3D datasets are supported (an easy way to do this is to inherit from [./GANDLF/models/modelBase](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/models/modelBase.py))

[Back To Top &uarr;](#table-of-contents)

## Adding Transformations

- Update [TorchIO](https://github.com/fepegar/torchio) version in [setup](https://github.com/CBICA/GaNDLF/blob/master/setup.py), if appropriate.
- Add transformation in [ImagesFromDataFrame Function](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/data/ImagesFromDataFrame.py), under `global_augs_dict`
- Ensure probability is used as input (not used for normalize or resample)

[Back To Top &uarr;](#table-of-contents)

### Adding Pre-processing functionality

- All transforms should be defined as [TorchIO Lambdas](https://torchio.readthedocs.io/transforms/others.html#lambda). For example, please see the threshold/clip functionality in the [./GANDLF/preprocessing.py](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/preprocessing.py) file.
- Define each option in the configuration file under the correct key (again, see threshold/clip as examples)

[Back To Top &uarr;](#table-of-contents)

## Add Training Functionality

- Update [Training Function](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/training_loop.py)
- Update [Training Manager](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/training_manager.py), if any training API has changed

[Back To Top &uarr;](#table-of-contents)

## Add Inference Functionality

- Update [Inference Function](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/inference_loop.py)
- Update [Inference Manager](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/inference_manager.py), if any inference API has changed

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
