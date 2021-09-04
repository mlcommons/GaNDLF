# GANDLF Models

## Adding a new model

- Follow example of `GANDLF.models.unet`.
- Define a new submodule under `GANDLF.models`, and define a class that inherits from `GANDLF.models.ModelBase`.
- Ensure that a forward pass is implemented.
- All parameters should be taken as input, with special parameters (for e.g., `residualConnections` for `unet`) should not be exposed to the parameters dict, and should be handled separately via another class.
    - For example, `GANDLF.models.unet.unet` has a `residualConnections` parameter, which is not exposed to the parameters dict, and a separate class `GANDLF.models.unet.resunet` is defined which enables this flag.
- Add the model's identifier to `GANDLF.models.__init__.global_model_dict` as appropriate.
- Call the new mode from the config using the `model` key.