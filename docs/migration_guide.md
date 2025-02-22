# Migration Guide

## Post-0.0.20 Release

The [0.0.20 release](https://github.com/mlcommons/GaNDLF/releases/tag/0.0.20) was the final release that supported the old way of using GaNDLF (i.e., `gandlf_run`). Instead, we now have a CLI that is more unified and based on modern CLI parsing (i.e., `gandlf run`). If you have been using version `0.0.20` or earlier, please follow this guide to move your experimental setup to the new CLI [[ref](https://github.com/mlcommons/GaNDLF/pull/845)].

### User-level Changes

#### Command Line Interfaces

- The CLI commands have been moved to use [`click`](https://click.palletsprojects.com/en/8.1.x/) for parsing the command line arguments. This means that the commands are now more user-friendly and easier to remember, as well as with added features like tab completion and type checks.
- All the commands that were previously available in as `gandlf_${functionality}` are now available as `gandlf ${functionality}` (i.e., replace the `_` with ` `). 
- The previous commands are still present, but they are deprecated and will be removed in a future release.

#### Configuration Files

- The main change is the use of the [Version package](https://github.com/keleshev/version) for systematic semantic versioning [[ref](https://github.com/mlcommons/GaNDLF/pull/841)]. 
- No change is needed if you are using a [stable version](https://docs.mlcommons.org/GaNDLF/setup/#install-from-package-managers).
- If you have installed GaNDLF [from source](https://docs.mlcommons.org/GaNDLF/setup/#install-from-sources) or using a [nightly build](https://docs.mlcommons.org/GaNDLF/setup/#install-from-package-managers), you will need to ensure that the `maximum` key under `version` in the configuration file contains the correct version number:
  - Either **including** the `-dev` identifier of the current version (e.g., if the current version is `0.X.Y-dev`, then the `maximum` key should be `0.X.Y-dev`).
  - Or **excluding** the `-dev` identifier of the current version, but increasing the version number by one on any level (e.g., if the current version is `0.X.Y-dev`, then the `maximum` key should be `0.X.Y`).

#### Use in HPC Environments

- If you are using GaNDLF in an HPC environment, you will need to update the job submission scripts to use the new CLI commands.
- The previous API required one to call the interpreter and the specific command (e.g., `${venv_gandlf}/bin/python gandlf_run`), while the new API requires one to call the GaNDLF command directly (e.g., `${venv_gandlf}/bin/gandlf run` or `${venv_gandlf}/bin/gandlf_run`).
- The [Slurm experiments template](https://github.com/IUCompPath/gandlf_experiments_template_slurm) has been appropriately updated to reflect this change.


### Developer-level Changes

#### Command Line Interfaces

- CLI entrypoints are now defined in the `GANDLF.entrypoints` module, which contains argument parsing (using both the old and new API structures).
- CLI entrypoint logic is now defined in the `GANDLF.cli` module, which only contains how the specific functionality is executed from an algorithmic perspective.
 - This is to ensure backwards API compatibility, and will **not** be removed.

#### Configuration Files

- GaNDLF's [`config_manager` module](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/config_manager.py) is now the primary way to manage configuration files.
- This is going to be updated to use [pydantic](https://docs.pydantic.dev/latest/) in the near future [[ref](https://github.com/mlcommons/GaNDLF/issues/758)].


## Post 0.1.3-dev release

The 0.1.3-dev release has introduced [Pytorch Lightning](https://lightning.ai/) as the primary orchestration framework for core GaNDLF functionality. From the user perspective, the changes are minimal, but the underlying architecture has been significantly updated.

### User-level Changes

#### Command Line Interfaces

- The need for specifying the used device from CLI command has been deprecated and will be removed in a future releases. The usage of given accelerator is now configured via configuration file. Thus, removal of `--device` \ `-d` argument from the CLI commands is recommended, as it does not have any effect anymore and will result in error in future releases.

#### Configuration Files

New parameters are introduced in the configuration file to control distributed training, mixed precision training, and automatic searching for optimal learning rate or batch size. The new parameters are:
- `accelerator` - defaults to `auto`, you can explicitly set it to `cpu`, `gpu`, or `tpu`. `auto` will try to use GPU if available, otherwise CPU.
- `precision` - defaults to `"32-true"`, `16`, `32`, `64`, `"64-true"`, `"32-true"`, `"16-mixed"`, `"bf16"`, `"bf16-mixed"`.
- `n_nodes` - defaults to `1`, set this to the number of compute nodes you wish to use. Ensure that the machines are properly interconnected, and that the `gandlf run` command is executed on all nodes.
- `devices` - defaults to `auto`, you can explicitly set it to a list of device ids to use. `auto` will try to use all available devices. Note that you need to launch `ganldf run` on each node number of times equal to the number of devices you want to use (one process per device).
- `strategy` - defaults to `auto`, specifies the strategy for distributed training. `auto` will try to use `ddp` (Data Distributed Parallel) if multiple GPUs are available. To read more about the strategies and their differences from currently supported Data Parallel strategy, refer to [Pytorch documentation](https://pytorch.org/tutorials/beginner/ddp_series_theory.html)
- `auto_lr_find` - defaults to `false`, if set to `true`, the learning rate will be automatically determined using the learning rate finder at the beginning of the training. To read more, refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html#lightning.pytorch.tuner.tuning.Tuner)
- `auto_batch_size_find` - defaults to `false`, if set to `true`, the batch size will be automatically determined using the batch size finder at the beginning of the training. The batch size will be set to the highest number possible that fits into the memory of the device. Note that batch size has effect only for the training dataloader, as the validation, test, and inference dataloaders use batch size of 1 due to the nature of the evaluation process implemented in GaNDLF. To read more, refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html#lightning.pytorch.tuner.tuning.Tuner)
- `num_workers_dataloader` - defaults to `1`, set this to the number of workers to use for the dataloader. This parameter will be used across all dataloaders in the training, validation, test, and inference. Note that the number of workers is limited by the number of CPU cores available on the machine. To read more, refer to [Pytorch documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- `pin_memory_dataloader` - defaults to `false`, if set to `true`, the dataloader will copy the data to the page-locked memory, which can increase the speed of data transfer to the GPU. To read more, refer to [Pytorch documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- `prefetch_factor_dataloader` - defaults to `2`, set this to the number of batches to prefetch per worker during data loading. To read more, refer to [Pytorch documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- `parallel compute command` - this command is deprecated and will be removed in future releases. The parallel compute command is now automatically determined based on the accelerator and the number of nodes and devices specified in the configuration file.

#### Use in HPC Environments

As mentioned in the previous section about the configuration file, distributed training is now done automatically by Pytorch Lightning. User has to only specify the number of nodes and devices in the configuration file, and ensure proper number of processes in each node is launched. The `gandlf run` command should be executed on each node number of times equal to the number of devices you want to use (one process per device). Example of script configuration to use when running with SLURM scheduler in the batch mode:
```bash
### When configuring SBATCH parameters, pay special attention to the following ones:
#SBATCH --nodes=2 # number of nodes
#SBATCH --ntasks-per-node=4 # number of tasks per node, should be equal to the number of devices you want to use per node
#SBATCH --cpus-per-task=10 # number of CPUs per task, should be equal to the number of workers in the dataloader per launched process
#SBATCH --gpus-per-task=1 # assign one GPU per task

### run the gandlf using srun to execute it n times on each node, where n is the number of tasks per node
srun gandlf run ....
```


### Developer-level Changes

#### Training and inference logic

Entire logic of training and inference is now handled by the LightningModule abstraction. The logic has been moved from previous implementation and is controlled by `GandlfLightningModule` in the `GANDLF/models` module. Using Lightning enforces implementation of specific method encapsulating the logic of each step of the training process. To readm more about the LightningModule, refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)

#### Data loading

Entire logic of preparing the dataset objects and configuring the dataloaders is now handled by the `GandlfTrainingDatamodule` and `GandlfInferenceDatamodule` in the `GANDLF/data` module. Some data objects that need to be constructed on the fly (during validation, test or inference) are handled in the previously described `GandlfLightningModule`, adhering to the previous GaNDLF logic. To read more about the Lightning DataModule, refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/common/datamodule.html)

#### Execution of the training and inference

Order of operations, as well as configuration of the compute environment, is now handled by the `Trainer` class from Lightning. To read more about the Trainer, refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html)