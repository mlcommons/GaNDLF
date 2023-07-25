# GANDLF Metrics

## Adding a new algorithm

- Define a new submodule under `GANDLF.data.metrics`.
- Fit the loss appropriately: `GANDLF.data.metrics.segmentation` or `GANDLF.data.metrics.regression`.
    - Metrics for classification go under `regression`.
- Ensure the new algorithm has a functional interface that accepts input in the format `algorithm(predicted, ground_truth, parameters)` and returns a Tensor value.
    - If the new algorithm is from a pre-defined package, put it under `GANDLF.data.metrics.wrap_${package_name}.py`.
- Add the algorithm's identifier to `GANDLF.data.metrics.__init__.global_metrics_dict` as appropriate.
- Call the new identified from the config using the `metrics` key.
