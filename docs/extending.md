# Extending GaNDLF

For any new feature, please ensure the corresponding option in the [sample configuration](https://github.com/CBICA/GaNDLF/blob/master/samples/sample_training.yaml) is added, so that others can review/use/extend it as needed.

## Environment

Before starting to work on the code-level on GaNDLF, please get the environment ready:

```bash
git clone https://github.com/CBICA/GaNDLF.git
cd GaNDLF
conda create -p ./venv python=3.6 -y
conda activate ./venv
conda install -c pytorch pytorch # 1.8.0 installs cuda 10.2 by default, personalize based on your cuda/driver availability via https://pytorch.org/get-started/locally/
## windows
# conda install -c sdvillal openslide -y
## linux
# conda install -c conda-forge libvips openslide -y
pip install -e .
```
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

## Adding Models

- Add model code in [./GANDLF/models/](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/models/)
- Update initialization in [./GANDLF/parameterParsing](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/parameterParsing.py)
- Ensure both 2D and 3D datasets are supported (an easy way to do this is to inherit from [./GANDLF/models/modelBase](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/models/modelBase.py))

## Adding Transformations

- Update [TorchIO](https://github.com/fepegar/torchio) version in [setup](https://github.com/CBICA/GaNDLF/blob/master/setup.py), if appropriate.
- Add transformation in [ImagesFromDataFrame Function](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/data/ImagesFromDataFrame.py), under `global_augs_dict`
- Ensure probability is used as input (not used for normalize or resample)

### Adding Pre-processing functionality

- All transforms should be defined as [TorchIO Lambdas](https://torchio.readthedocs.io/transforms/others.html#lambda). For example, please see the threshold/clip functionality in the [./GANDLF/preprocessing.py](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/preprocessing.py) file.
- Define each option in the configuration file under the correct key (again, see threshold/clip as examples)

## Add Training Functionality

- Update [Training Function](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/training_loop.py)
- Update [Training Manager](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/training_manager.py), if any training API has changed

## Add Inference Functionality

- Update [Inference Function](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/inference_loop.py)
- Update [Inference Manager](https://github.com/CBICA/GaNDLF/blob/master/GANDLF/inference_manager.py), if any inference API has changed

## Update Tests

Once you have made changes to functionality, it is imperative that the unit tests be updated to cover the new code. Please see the [full testing suite](https://github.com/CBICA/GaNDLF/blob/master/testing/test_full.py) for details and examples.

## Run Tests

Once you have the virtual environment set up, tests can be run using the following command:
```powershell
pytest --device cuda # can be cuda or cpu, defaults to cpu
```

### Code coverage

The code coverage for the tests can be obtained by the following command:
```powershell
coverage run -m pytest --device cuda; coverage report -m
```
