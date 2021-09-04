# GANDLF Scheduler

## Adding a new algorithm

- Define a new submodule under `GANDLF.scheduler`.
- Ensure that the new algorithm is wrapped in a function which returns a scheduler, by following one of the examples in `GANDLF.scheduler.triangle`.
- Add the algorithm's identifier to `GANDLF.scheduler.__init__.global_schedulers_dict` as appropriate.
- Call the new algorithm from the config using the `scheduler` key.