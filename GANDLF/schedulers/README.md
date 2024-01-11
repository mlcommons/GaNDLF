# GANDLF schedulers

## Adding a new algorithm

- For a scheduler defined in PyTorch [[ref](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)], update the `GANDLF.schedulers.wrap_torch.py` submodule.
- For a custom scheduler, create a new submodule called `GANDLF.schedulers.${awesome_optimizer}.py`. Ensure that it inherits from PyTorch's base optimizer class [[ref](https://pytorch.org/docs/stable/optim.html#base-class)]. Follow the example of [`torch.optim.lr_scheduler.LinearLR`](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LinearLR) as an example.
- If a new dependency needs to be used, update GaNDLF's [`setup.py`](https://github.com/mlcommons/GaNDLF/blob/master/setup.py) with the new requirement.
  - Define a new submodule under `GANDLF.schedulers` as `GANDLF.schedulers.wrap_${package_name}.py`.
  - Ensure that the new algorithm is wrapped in a function which returns an object with the PyTorch optimizer type. Use any of the optimizers in `GANDLF.schedulers.wrap_torch.py` as an example.
- Add the algorithm's identifier to `GANDLF.schedulers.__init__.global_optimizer_dict` with an appropriate key.
- Call the new algorithm from the config using the `scheduler` key.
- [Update the tests!](https://mlcommons.github.io/GaNDLF/extending/#update-tests)https://mlcommons.github.io/GaNDLF/extending/#update-tests
