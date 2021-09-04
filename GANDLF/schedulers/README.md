# GANDLF schedulers

## Adding a new algorithm

- Define a new submodule under `GANDLF.schedulers`.
- Ensure that the new algorithm is wrapped in a function which returns a schedulers, by following one of the examples in `GANDLF.schedulers.triangle`.
    - If the new function is from a pre-defined package, put it under `GANDLF.schedulers.wrap_${package_name}.py`.
- Add the algorithm's identifier to `GANDLF.schedulers.__init__.global_schedulerss_dict` as appropriate.
- Call the new algorithm from the config using the `scheduler` key.