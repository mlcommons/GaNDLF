# GANDLF Post-processing

## Adding a new algorithm

- Define a new submodule under `GANDLF.data.post_process`, or update an existing one.
- Add the algorithm's identifier to `GANDLF.data.post_process.__init__.global_postprocessing_dict` as appropriate.
- Call the new algorithm from the config using the `data_postprocessing` key.
- Care should be taken that the post-processing steps should only be called during the `"save_output"` routine of `GANDLF.compute.forward_pass`, so that validation results do not get tainted by any post-processing.
- If the new algorithm is to be applied after reverse one-hot encoding, then append the key to `GANDLF.data.post_process.__init__.postprocessing_after_reverse_one_hot_encoding`. Ensure that these algorithms return `numpy.ndarray`.