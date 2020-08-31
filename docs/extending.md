# Extending GANDLF

For any new feature, please ensure the corresponding option in the [sample configuration](../samples/sample_training.yaml) is added, so that others can review/use/extend it as needed.

## Architecture

- Command-line parsing: [gandlf_run](../gandlf_run)
- Parameters from [training configuration](../samples/sample_training.yaml) getting passed as dict via [parameter parser](../GANDLF/parseConfig.py)
- [Training Manager](../GANDLF/training_manager.py): 
  - Handles k-fold training 
  - Main entry point from CLI
- [Training Function](../GANDLF/training_loop.py): 
  - Performs actual training
- [Inference Manager](../GANDLF/inference_manager.py): 
  - Handles inference functionality 
  - Main entry point from CLI
- [Inference Function](../GANDLF/inference_loop.py): 
  - Performs actual inference

## Adding Models

- Add model code in [./GANDLF/models/](../GANDLF/models/)
- Ensure both 2D and 3D models are supported

## Adding Transformations

- Update [TorchIO](https://github.com/fepegar/torchio) version in [setup](../setup.py), if appropriate.
- Add transformation in [ImagesFromDataFrame Function](../GANDLF/data/ImagesFromDataFrame.py), under `global_augs_dict`
- Ensure probability is used as input (not used for normalize or resample)

## Add Training Functionality

- Update [Training Function](../GANDLF/training_loop.py)
- Update [Training Manager](../GANDLF/training_manager.py), if any training API has changed

## Add Inference Functionality

- Update [Inference Function](../GANDLF/inference_loop.py)
- Update [Inference Manager](../GANDLF/inference_manager.py), if any inference API has changed
