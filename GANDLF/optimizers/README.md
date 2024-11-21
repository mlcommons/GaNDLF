# GANDLF Optimizers

## Adding a new algorithm

- For an optimizer defined in PyTorch [[ref](https://pytorch.org/docs/stable/optim.html#algorithms)], update the `GANDLF.optimizers.wrap_torch.py` submodule.
- For a custom optimizer, create a new submodule called `GANDLF.optimizers.${awesome_optimizer}.py`.
- For a third-party optimizer (i.e., where the code is available from an external source/repository):
  - Add the relevant code under the `GANDLF.optimizers.thirdparty` submodule. 
  - Add a wrapper which takes in GaNDLF's `parameter` dictionary as input and creates a `torch.optim.Optimizer` object as output.
  - Add the wrapper to the `GANDLF.optimizers.thirdparty.__init__.py` so that it can be called from `GANDLF.optimizers.__init__.py`.
  - See `GANDLF.optimizers.thirdparty.adopt.py` as an example.
- If a new dependency needs to be used, update GaNDLF's [`setup.py`](https://github.com/mlcommons/GaNDLF/blob/master/setup.py) with the new requirement.
  - Define a new submodule under `GANDLF.optimizers` as `GANDLF.optimizers.wrap_${package_name}.py`.
  - Ensure that the new algorithm is wrapped in a function which returns an object with the PyTorch optimizer type. Use any of the optimizers in `GANDLF.optimizers.wrap_torch.py` as an example.
- Add the algorithm's identifier to `GANDLF.optimizers.__init__.global_optimizer_dict` with an appropriate key.
- Call the new algorithm from the config using the `optimizer` key.
- [If appropriate, please update the tests!](https://mlcommons.github.io/GaNDLF/extending/#update-tests)https://mlcommons.github.io/GaNDLF/extending/#update-tests
- All wrappers should return the type `from torch.optim.optimizer.Optimizer`.