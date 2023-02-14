# mlcube

This directory is a template for creating MLCube directories that GaNDLF can deploy.

The workspace contains a sample "channelIDs.yml" that can be passed to the construct_csv task and a "config.yml" example used for training/inference.
However, generally, the workspace will be populated with a user's own files. 

It is recommended that you distribute at least "config.yml" and a "mlcube.yaml" alongside your MLCube.
However, the "recover_config" task allows users to recover a usable config.yml from the MLCube itself, if needed.
