# GANDLF Optimizers

## Adding a new algorithm

- For an optimizer defined in PyTorch [[ref](https://pytorch.org/docs/stable/optim.html#algorithms)], update the `GANDLF.optimizers.wrap_torch.py` submodule.
- For a custom optimizer, create a new submodule called `GANDLF.optimizers.${awesome_optimizer}.py`. Ensure that it inherits from PyTorch's base optimizer class [[ref](https://pytorch.org/docs/stable/optim.html#base-class)]
- If a new dependency needs to be used, update GaNDLF's [`setup.py`](https://github.com/mlcommons/GaNDLF/blob/master/setup.py) with the new requirement.
  - Define a new submodule under `GANDLF.optimizers` as `GANDLF.optimizers.wrap_${package_name}.py`.
  - Ensure that the new algorithm is wrapped in a function which returns an object with the PyTorch optimizer type. Use any of the optimizers in `GANDLF.optimizers.wrap_torch.py` as an example.
- Add the algorithm's identifier to `GANDLF.optimizers.__init__.global_optimizer_dict` with an appropriate key.
- Call the new algorithm from the config using the `optimizer` key.
- [Update the tests!](https://mlcommons.github.io/GaNDLF/extending/#update-tests)https://mlcommons.github.io/GaNDLF/extending/#update-tests
