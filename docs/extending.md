# Extending GANDLF

For any new feature, please ensure the corresponding option in the [sample configuration](../samples/sample_training.yaml) is added, so that others can review/use/extend it as needed.

## Adding Models

- Add model code in [./GANDLF/models/](../GANDLF/models/)
- Ensure both 2D and 3D models are supported

## Adding Transformations

- Update [TorchIO](https://github.com/fepegar/torchio) version in [setup](../setup.py), if appropriate.
- Add transformation in [ImagesFromDataFrame Function](../GANDLF/data/ImagesFromDataFrame.py), under `global_augs_dict`
- Ensure probability is used as input (not used for normalize or resample)