# GANDLF Preprocessing

## Adding a new algorithm

- Define a new submodule under `GANDLF.data.preprocessing`.
- Ensure that the new algorithm is wrapped in a class, by following the example of `GANDLF.data.preprocessing.non_zero_normalize.NonZeroNormalizeOnMaskedRegion`.
    - This uses a `torchio.Transform` as a base class. 
    - The class name is the algorithm's identifier for GaNDLF.
    - Add it as a new submodule.
    - Wrap the new algorithm in a `_transform` function, which makes it easier to define algorithmic properties (see `GANDLF.data.preprocessing.threshold_and_clip` as an example, where the same class is used for both threshold and clip functionalities) If new algorithm can be wrapped in a function:
    - Add function in the new submodule.
    - The `_transform` function is the algorithm's identifier for GaNDLF.
- Add the algorithm's identifier to `GANDLF.data.preprocessing.__init__.global_preprocessing_dict` as appropriate.
- Call the new algorithm from the config using the `data_preprocessing` key.