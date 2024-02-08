## Environment

Before starting to work on the code-level on GaNDLF, please follow the [instructions to install GaNDLF from sources](./setup.md#install-from-sources). Once that's done, please verify the installation using the following command:

```bash
# continue from previous shell
(venv_gandlf) $> 
# you should be in the "GaNDLF" git repo
(venv_gandlf) $> python ./gandlf_verifyInstall
```


## Submodule flowcharts

- The following flowcharts are intended to provide a high-level overview of the different submodules in GaNDLF. 
- Navigate to the `README.md` file in each submodule folder for details.

## Overall Architecture

- Command-line parsing: [gandlf_run](https://github.com/mlcommons/GaNDLF/blob/master/gandlf_run)
- Parameters from [training configuration](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_all_options.yaml) get passed as a `dict` via the [config manager](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/config_manager.py)
- [Training Manager](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/training_manager.py): 
    - Handles k-fold training 
    - Main entry point from CLI
- [Training Function](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/compute/training_loop.py): 
    - Performs actual training
- [Inference Manager](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/inference_manager.py): 
    - Handles inference functionality 
    - Main entry point from CLI
- [Inference Function](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/compute/inference_loop.py): 
    - Performs actual inference


## Dependency Management

To update/change/add a dependency in [setup](https://github.com/mlcommons/GaNDLF/blob/master/setup.py), please ensure **at least** the following conditions are met:

- The package is being [actively maintained](https://opensource.com/life/14/1/evaluate-sustainability-open-source-project).
- The new dependency is being testing against the **minimum python version** supported by GaNDLF (see the `python_requires` variable in [setup](https://github.com/mlcommons/GaNDLF/blob/master/setup.py)).
- It does not clash with any existing dependencies.

## Adding Models

- For details, please see [README for `GANDLF.models` submodule](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/models/Readme.md).
- [Update Tests](#update-tests)


## Adding Augmentation Transformations

- Update or add dependency in [setup](https://github.com/mlcommons/GaNDLF/blob/master/setup.py), if appropriate.
- Add transformation to `global_augs_dict`, defined in [`GANDLF/data/augmentation/__init__.py`](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/data/augmentation/__init__.py)
- Ensure probability is used as input; probability is not used for any [preprocessing operations](https://github.com/mlcommons/GaNDLF/tree/master/GANDLF/data/preprocessing)
- For details, please see [README for `GANDLF.data.augmentation` submodule](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/data/augmentation/README.md).
- [Update Tests](#update-tests)


## Adding Preprocessing functionality

- Update or add dependency in [setup](https://github.com/mlcommons/GaNDLF/blob/master/setup.py), if appropriate; see section on [Dependency Management](#dependency-management) for details.
- All transforms should be defined by inheriting from `torchio.transforms.intensity_transform.IntensityTransform`. For example, please see the threshold/clip functionality in the [`GANDLF/data/preprocessing/threshold_and_clip.py`](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/data/preprocessing/threshold_and_clip.py) file.
- Define each option in the configuration file under the correct key (again, see threshold/clip as examples)
- Add transformation to `global_preprocessing_dict`, defined in [`GANDLF/data/preprocessing/__init__.py`](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/data/preprocessing/__init__.py)
- For details, please see [README for `GANDLF.data.preprocessing` submodule](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/data/preprocessing/README.md).
- [Update Tests](#update-tests)


## Adding Training Functionality

- Update [Training Function](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/compute/training_loop.py)
- Update [Training Manager](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/training_manager.py), if any training API has changed
- [Update Tests](#update-tests)


## Adding Inference Functionality

- Update [Inference Function](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/compute/inference_loop.py)
- Update [Inference Manager](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/inference_manager.py), if any inference API has changed
- [Update Tests](#update-tests)


## Update parameters

For any new feature, please ensure the corresponding option in the [sample configuration](https://github.com/mlcommons/GaNDLF/blob/master/samples/sample_training.yaml) is added, so that others can review/use/extend it as needed.


## Update Tests

Once you have made changes to functionality, it is imperative that the unit tests be updated to cover the new code. Please see the [full testing suite](https://github.com/mlcommons/GaNDLF/blob/master/testing/test_full.py) for details and examples.


## Run Tests

Once you have the virtual environment set up, tests can be run using the following command:

```bash
# continue from previous shell
(venv_gandlf) $> pytest --device cuda # can be cuda or cpu, defaults to cpu
```

Any failures will be reported in the file [`${GaNDLF_HOME}/testing/failures.log`](https://github.com/mlcommons/GaNDLF/blob/5030ff83a38947c1583b58a08598308886ee9a0a/testing/conftest.py#L25).


### Code coverage

The code coverage for the tests can be obtained by the following command:

```powershell
bash
# continue from previous shell
(venv_gandlf) $> coverage run -m pytest --device cuda; coverage report -m
```

