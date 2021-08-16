# GANDLF Losses

## Adding a new algorithm

- Define a new submodule under `GANDLF.data.losses`.
- Fit the loss appropriately: `GANDLF.data.losses.segmentation` or `GANDLF.data.losses.regression`.
    - Losses for classification go under `regression`.
- Ensure the new algorithm has a functional interface that accepts input in the format `algorithm(predicted, ground_truth, parameters)` and returns a Tensor value.
    - If the new algorithm is from a pre-defined package, put it under `GANDLF.data.losses.wrap_${package_name}.py`.
- Add the algorithm's identifier to `GANDLF.data.losses.__init__.global_losses_dict` as appropriate.
- Call the new identified from the config using the `loss_function` key.
