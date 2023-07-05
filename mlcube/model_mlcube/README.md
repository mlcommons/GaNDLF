# mlcube

This directory is a template for creating MLCube directories that GaNDLF can deploy.

The `workspace` directory contains a sample `channelIDs.yml` that can be passed to the [`gandlf_constructCSV`](https://mlcommons.github.io/GaNDLF/usage/#constructing-the-data-csv) task and a `config.yml` example used for training/inference. However, generally, the workspace will be populated with a user's own files. 

It is recommended that you distribute at least `config.yml` and a `mlcube.yaml` alongside your MLCube.
However, the `gandlf_recoverConfig` task allows users to recover a usable configuration from the MLCube itself, if needed.
