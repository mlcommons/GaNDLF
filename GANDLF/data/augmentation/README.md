# GANDLF Augmentation

## Adding a new algorithm

- Define a new submodule under `GANDLF.data.augmentation`.
- If new algorithm is a class, follow principle of adding torchio transforms. The class name is the algorithm's identifier for GaNDLF.
- If new algorithm can be wrapped in a function:
    - Add function in new submodule.
    - Ensure it returns a `torch.Tensor`.
    - The function name is the algorithm's identifier for GaNDLF.
- Add the algorithm's identifier to `GANDLF.data.augmentation.global.global_augs_dict` as appropiate.
- Call the new identified from the config using the `data_augmentation` key.