# Migration Guide

The [0.0.20 release](https://github.com/mlcommons/GaNDLF/releases/tag/0.0.20) was the final release that supported the old way of using GaNDLF (i.e., `gandlf_run`). Instead, we now have a CLI that is more unified and based on modern CLI parsing (i.e., `gandlf run`). If you have been using version `0.0.20` or earlier, please follow this guide to move your experimental setup to the new CLI [[ref](https://github.com/mlcommons/GaNDLF/pull/845)].

## User-level Changes

### Command Line Interfaces

- The CLI commands have been moved to use [`click`](https://click.palletsprojects.com/en/8.1.x/) for parsing the command line arguments. This means that the commands are now more user-friendly and easier to remember, as well as with added features like tab completion and type checks.
- All the commands that were previously available in as `gandlf_${functionality}` are now available as `gandlf ${functionality}` (i.e., replace the `_` with ` `). 
- The previous commands are still present, but they are deprecated and will be removed in a future release.

### Configuration Files

- The main change is the use of the [Version package](https://github.com/keleshev/version) for systematic semantic versioning [[ref](https://github.com/mlcommons/GaNDLF/pull/841)]. 
- No change is needed if you are using a [stable version](https://docs.mlcommons.org/GaNDLF/setup/#install-from-package-managers).
- If you have installed GaNDLF [from source](https://docs.mlcommons.org/GaNDLF/setup/#install-from-sources) or using a [nightly build](https://docs.mlcommons.org/GaNDLF/setup/#install-from-package-managers), you will need to ensure that the `maximum` key under `version` in the configuration file contains the correct version number:
  - Either **including** the `-dev` identifier of the current version (e.g., if the current version is `0.X.Y-dev`, then the `maximum` key should be `0.X.Y-dev`).
  - Or **excluding** the `-dev` identifier of the current version, but increasing the version number by one on any level (e.g., if the current version is `0.X.Y-dev`, then the `maximum` key should be `0.X.Y`).

### Use in HPC Environments

- If you are using GaNDLF in an HPC environment, you will need to update the job submission scripts to use the new CLI commands.
- The previous API required one to call the interpreter and the specific command (e.g., `${venv_gandlf}/bin/python gandlf_run`), while the new API requires one to call the GaNDLF command directly (e.g., `${venv_gandlf}/bin/gandlf run` or `${venv_gandlf}/bin/gandlf_run`).
- The [Slurm experiments template](https://github.com/IUCompPath/gandlf_experiments_template_slurm) has been appropriately updated to reflect this change.


## Developer-level Changes

### Command Line Interfaces

- CLI entrypoints are now defined in the `GANDLF.entrypoints` module, which contains argument parsing (using both the old and new API structures).
- CLI entrypoint logic is now defined in the `GANDLF.cli` module, which only contains how the specific functionality is executed from an algorithmic perspective.
 - This is to ensure backwards API compatibility, and will **not** be removed.

### Configuration Files

- GaNDLF's [`config_manager` module](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/config_manager.py) is now the primary way to manage configuration files.
- This is going to be updated to use [pydantic](https://docs.pydantic.dev/latest/) in the near future [[ref](https://github.com/mlcommons/GaNDLF/issues/758)].
