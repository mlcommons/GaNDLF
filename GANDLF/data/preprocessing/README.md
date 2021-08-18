# GANDLF Preprocessing

## Adding a new algorithm

- Define a new submodule under `GANDLF.data.preprocessing`.
- If new algorithm is a class, follow principle of `GANDLF.data.preprocessing.non_zero_normalize.NonZeroNormalizeOnMaskedRegion` which uses a `torchio.Transform` as a base class. The class name is the algorithm's identifier for GaNDLF.
- If new algorithm can be wrapped in a function:
    - Add function in new submodule.
    - Wrap it in a `_transform` function using `torchio.Lambda` as shown in `GANDLF.data.preprocessing.threshold_and_clip.threshold_transform`.
    - The `_transform` function is the algorithm's identifier for GaNDLF.
- Add the algorithm's identifier to `GANDLF.data.preprocessing.__init__.global_preprocessing_dict` as appropriate.
- Call the new identified from the config using the `data_preprocessing` key.