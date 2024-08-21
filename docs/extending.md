## Environment

Before starting to work on the code-level on GaNDLF, please follow the [instructions to install GaNDLF from sources](./setup.md#install-from-sources). Once that's done, please verify the installation using the following command:

```bash
# continue from previous shell
(venv_gandlf) $> 
# you should be in the "GaNDLF" git repo
(venv_gandlf) $> gandlf verify-install
```


## Submodule flowcharts

- The following flowcharts are intended to provide a high-level overview of the different submodules in GaNDLF. 
- Navigate to the `README.md` file in each submodule folder for details.

## Overall Architecture

- Command-line parsing: [gandlf run](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/entrypoints/run.py)
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

## Adding new CLI command
Example: `gandlf config-generator` [CLI command](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/entrypoints/config_generator.py)
- Implement function and wrap it with `@click.command()` + `@click.option()`
- Add it to `cli_subommands` [dict](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/entrypoints/subcommands.py)
The command would be available under `gandlf your-subcommand-name` CLI command.

## Update parameters

For any new feature, please ensure the corresponding option in the [sample configuration](https://github.com/mlcommons/GaNDLF/blob/master/samples/sample_training.yaml) is added, so that others can review/use/extend it as needed.


## Update Tests

Once you have made changes to functionality, it is imperative that the unit tests be updated to cover the new code. Please see the [full testing suite](https://github.com/mlcommons/GaNDLF/blob/master/testing/test_full.py) for details and examples.


## Run Tests

### Prerequisites

There are two types of tests: unit tests for GaNDLF code, which tests the functionality, and integration tests for deploying and running mlcubes. Some additional steps are required for running tests:

1. Ensure that the install optional dependencies [[ref](https://mlcommons.github.io/GaNDLF/setup/#optional-dependencies)] have been installed.
2. Tests are using [sample data](https://drive.google.com/uc?id=1c4Yrv-jnK6Tk7Ne1HmMTChv-4nYk43NT), which gets downloaded and prepared automatically when you run unit tests. Prepared data is stored at `${GaNDLF_root_dir}/testing/data/` folder. However, you may want to download & explore data by yourself.

### Unit tests

Once you have the virtual environment set up, tests can be run using the following command:

```bash
# continue from previous shell
(venv_gandlf) $> pytest --device cuda # can be cuda or cpu, defaults to cpu
```

Any failures will be reported in the file [`${GANDLF_HOME}/testing/failures.log`](https://github.com/mlcommons/GaNDLF/blob/5030ff83a38947c1583b58a08598308886ee9a0a/testing/conftest.py#L25).

### Integration tests

All integration tests are combined to one shell script:

```shell
# it's assumed you are in `GaNDLF/` repo root directory
cd testing/
./test_deploy.sh
```

### Code coverage

The code coverage for the unit tests can be obtained by the following command:

```powershell
bash
# continue from previous shell
(venv_gandlf) $> coverage run -m pytest --device cuda; coverage report -m
```
## Logging

### Use loggers instead of print
We use the native `logging` [library](https://docs.python.org/3/library/logging.html) for logs management. This gets automatically configured when GaNDLF gets launched. So, if you are extending the code, please use loggers instead of prints.

Here is an example how `root logger` can be used
```
def my_new_cool_function(df: pd.DataFrame):
    logging.debug("Message for debug file only")
    logging.info("Hi GaNDLF user, I greet you in the CLI output")
    logging.error(f"A detailed message about any error if needed. Exception: {str(e)}, params: {params}, df shape: {df.shape}")
    # do NOT use normal print statements
    # print("Hi GaNDLF user!")
```

Here is an example how logger can be used:

```
def my_new_cool_function(df: pd.DataFrame):
    logger = logging.getLogger(__name__)  # you can use any your own logger name or just pass a current file name
    logger.debug("Message for debug file only")
    logger.info("Hi GaNDLF user, I greet you in the CLI output")
    logger.error(f"A detailed message about any error if needed. Exception: {str(e)}, params: {params}, df shape: {df.shape}")
    # print("Hi GaNDLF user!")  # don't use prints please.
```


### What and where is logged

GaNDLF logs are splitted into multiple parts:
- CLI output: only `info` messages are shown here
- debug file: all messages are shown 
- stderr: display `warning`, `error`, or `critical` messages

By default, the logs are saved in the `/tmp/.gandlf` dir.
The logs are **saved** in the path that is defined by the '--log-file' parameter in the CLI commands.




Example of log message
```
#format: "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
2024-07-03 13:05:51,642 - root - DEBUG - GaNDLF/GANDLF/entrypoints/anonymizer.py:28 - input_dir='.'
```

### Create your own logger
You can create and configure your own logger by updating the file `GANDLF/logging_config.yaml`.



