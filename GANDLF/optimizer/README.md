# GANDLF Preprocessing

## Adding a new algorithm

- Define a new submodule under `GANDLF.optimizer`.
- Ensure that the new algorithm is wrapped in a function which returns a scheduler, by following one of the examples in `GANDLF.optimizer.sgd`.
    - If the new function is from a pre-defined package, put it under `GANDLF.optimizer.wrap_${package_name}.py`.
- Add the algorithm's identifier to `GANDLF.optimizer.__init__.global_optimizer_dict` as appropriate.
- Call the new algorithm from the config using the `optimizer` key.