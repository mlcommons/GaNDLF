This page contains answers to frequently asked questions about GaNDLF.

## Table of Contents
- [Table of Contents](#table-of-contents)
  - [Why do I get the error `pkg_resources.DistributionNotFound: The 'GANDLF' distribution was not found`?](#why-do-i-get-the-error-pkg_resourcesdistributionnotfound-the-gandlf-distribution-was-not-found)
  - [Where do I start?](#where-do-i-start)
  - [Why is GaNDLF not working?](#why-is-gandlf-not-working)
  - [Which parts of a GaNDLF configuration are customizable?](#which-parts-of-a-gandlf-configuration-are-customizable)
  - [Can I run GaNDLF on a high performance computing (HPC) cluster?](#can-i-run-gandlf-on-a-high-performance-computing-hpc-cluster)
  - [How can I track the per-epoch training performance?](#how-can-i-track-the-per-epoch-training-performance)
  - [How can I resume training from a previous checkpoint?](#how-can-i-resume-training-from-a-previous-checkpoint)
  - [How can I update GaNDLF?](#how-can-i-update-gandlf)
  - [What I have another question?](#what-i-have-another-question)

### Why do I get the error `pkg_resources.DistributionNotFound: The 'GANDLF' distribution was not found`?

This means that GaNDLF was not installed correctly. Please ensure you have followed the [installation guide](https://cbica.github.io/GaNDLF/setup) properly.

[Back To Top &uarr;](#table-of-contents)

### Where do I start?

The [usage](https://cbica.github.io/GaNDLF/usage) guide is fairly comprehensive and provides a good starting point. If you have any questions, please feel free to [post a support request](https://github.com/CBICA/GaNDLF/issues/new?assignees=&labels=&template=--questions-help-support.md&title=), and we will do our best to address it ASAP.

[Back To Top &uarr;](#table-of-contents)

### Why is GaNDLF not working?

Verify that [the installation](https://cbica.github.io/GaNDLF/setup) has been done correctly by running `python ./gandlf_verifyInstall` after activating the correct virtual environment. If you are still having issues, please feel free to [post a support request](https://github.com/CBICA/GaNDLF/issues/new?assignees=&labels=&template=--questions-help-support.md&title=), and we will do our best to address it ASAP.

[Back To Top &uarr;](#table-of-contents)

### Which parts of a GaNDLF configuration are customizable?

Virtually all of it! For more details, please see the [usage](https://cbica.github.io/GaNDLF/usage) guide and our extensive [samples](https://github.com/CBICA/GaNDLF/tree/master/samples). All available options are documented in the [config_all_options.yaml file](https://github.com/CBICA/GaNDLF/blob/master/samples/config_all_options.yaml).

[Back To Top &uarr;](#table-of-contents)

### Can I run GaNDLF on a high performance computing (HPC) cluster?

YES, we have run GaNDLF on an SGE cluster to great success. Please [post a question](https://github.com/CBICA/GaNDLF/issues/new?assignees=&labels=&template=--questions-help-support.md&title=) with more details such as the type of scheduler, and so on, and we will do our best to address it.

[Back To Top &uarr;](#table-of-contents)

### How can I track the per-epoch training performance?

Yes, look for `logs_*.csv` files in the output directory. It should be arranged in accordance with the cross-validation configuration, and should contain separate files for training/validation/testing data, along with the values for all requested performance metrics.

[Back To Top &uarr;](#table-of-contents)

### How can I resume training from a previous checkpoint?

GaNDLF allows you to resume training from a previous checkpoint in 2 ways:
- By using the `--resume` CLI parameter in `gandlf_run`, only the model weights and state dictionary will be preserved, but parameters and data are taken from the new options in the CLI. This is helpful when you are updated the training data or **some** compatible options in the parameters.
- If both `--resume` and `--reset` are `False` in `gandlf_run`, the model weights, state dictionary, and all previously saved information (parameters, training/validation/testing data) is used to resume training.

### How can I update GaNDLF?

If you have performed `git clone` [during installation](https://cbica.github.io/GaNDLF/setup), then you will need to do `git pull` to get the latest master of GaNDLF. Follow this up with `pip install -e .` after activating the appropriate virtual environment to ensure the updates get passed through.

[Back To Top &uarr;](#table-of-contents)

### What I have another question?

Please [post a support request](https://github.com/CBICA/GaNDLF/issues/new?assignees=&labels=&template=--questions-help-support.md&title=).

[Back To Top &uarr;](#table-of-contents)
