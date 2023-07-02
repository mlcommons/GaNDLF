This page contains answers to frequently asked questions about GaNDLF.


### Where do I start?

The [usage guide](https://mlcommons.github.io/GaNDLF/usage) provides a good starting point for you to understand the application of GaNDLF. If you have any questions, please feel free to [post a support request](https://github.com/mlcommons/GaNDLF/issues/new?assignees=&labels=&template=--questions-help-support.md&title=), and we will do our best to address it ASAP.

### Why do I get the error `pkg_resources.DistributionNotFound: The 'GANDLF' distribution was not found`?

This means that GaNDLF was not installed correctly. Please ensure you have followed the [installation guide](https://mlcommons.github.io/GaNDLF/setup) properly.

### Why is GaNDLF not working?

Verify that [the installation](https://mlcommons.github.io/GaNDLF/setup) has been done correctly by running `python ./gandlf_verifyInstall` after activating the correct virtual environment. If you are still having issues, please feel free to [post a support request](https://github.com/mlcommons/GaNDLF/issues/new?assignees=&labels=&template=--questions-help-support.md&title=), and we will do our best to address it ASAP.

### Which parts of a GaNDLF configuration are customizable?

Virtually all of it! For more details, please see the [usage](https://mlcommons.github.io/GaNDLF/usage) guide and our extensive [samples](https://github.com/mlcommons/GaNDLF/tree/master/samples). All available options are documented in the [config_all_options.yaml file](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_all_options.yaml).

### Can I run GaNDLF on a high performance computing (HPC) cluster?

Yes, GaNDLF has successfully been run on an SGE cluster and another managed using Kubernetes. Please [post a question](https://github.com/mlcommons/GaNDLF/issues/new?assignees=&labels=&template=--questions-help-support.md&title=) with more details such as the type of scheduler, and so on, and we will do our best to address it.

### How can I track the per-epoch training performance?

Yes, look for `logs_*.csv` files in the output directory. It should be arranged in accordance with the cross-validation configuration. Furthermore, it  should contain separate files for each data cohort, i.e., training/validation/testing, along with the values for all requested performance metrics, which are defined per problem type.

### Why are my compute jobs failing with excess RAM usage?

If you have `data_preprocessing` enabled, GaNDLF will load all of the resized images as tensors into memory. Depending on your dataset (resolution, size, number of modalities), this can lead to high RAM usage. To avoid this, you can enable the memory saver mode by enabling the flag `memory_save_mode` in the configuration. This will write the resized images into disk.

### How can I resume training from a previous checkpoint?

GaNDLF allows you to resume training from a previous checkpoint in 2 ways:
- By using the `--resume` CLI parameter in `gandlf_run`, only the model weights and state dictionary will be preserved, but parameters and data are taken from the new options in the CLI. This is helpful when you are updated the training data or **some** compatible options in the parameters.
- If both `--resume` and `--reset` are `False` in `gandlf_run`, the model weights, state dictionary, and all previously saved information (parameters, training/validation/testing data) is used to resume training.

### How can I update GaNDLF?

- If you have [installed from pip](./setup.md), then you can simply run `pip install --upgrade gandlf` to get the latest version of GaNDLF, or if you are interested in the nightly builds, then you can run `pip install --upgrade --pre gandlf`.
- If you have performed [installation from sources](./extending.md), then you will need to do `git pull` from the base `GaNDLF` directory to get the latest master of GaNDLF. Follow this up with `pip install -e .` after activating the appropriate virtual environment to ensure the updates get passed through.

### How can I perform federated learning of my GaNDLF model?

Please see https://mlcommons.github.io/GaNDLF/usage/#federating-your-model-using-openfl.

### How can I perform federated evaluation of my GaNDLF model?

Please see https://mlcommons.github.io/GaNDLF/usage/#federating-your-model-evaluation-using-medperf.

### What if I have another question?

Please [post a support request](https://github.com/mlcommons/GaNDLF/issues/new?assignees=&labels=&template=--questions-help-support.md&title=).