# GANDLF Augmentation

## Adding a new algorithm

- Define a new submodule under `GANDLF.data.augmentation`.
- If new algorithm is a class, follow principle of adding torchio transforms. 
- If new algorithm can be wrapped in a function:
    - Add function in new submodule.
    - Ensure it returns a `torch.Tensor`.
    - If the new function is from a pre-defined package, put it under `GANDLF.data.augmentation.wrap_${package_name}.py`.
- Wrap the new algorithm in a `_transform` function as shown in `GANDLF.data.augmentation.crop_external_zero_planes.crop_external_zero_planes_transform`. This is the algorithm's identifier for GaNDLF.
- Add the algorithm's identifier to `GANDLF.data.augmentation.__init__.global_augs_dict` as appropriate.
- Call the new algorithm from the config using the `data_augmentation` key.