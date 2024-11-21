from .wrap_torch import (
    sgd,
    asgd,
    adam,
    adamw,
    adamax,
    # sparseadam,
    rprop,
    adadelta,
    adagrad,
    rmsprop,
    radam,
    nadam,
)

from .wrap_monai import novograd_wrapper

from .thirdparty import ademamix_wrapper, lion_wrapper, adopt_wrapper

global_optimizer_dict = {
    "sgd": sgd,
    "asgd": asgd,
    "adam": adam,
    "adamw": adamw,
    "adamax": adamax,
    # "sparseadam": sparseadam,
    "rprop": rprop,
    "adadelta": adadelta,
    "adagrad": adagrad,
    "rmsprop": rmsprop,
    "radam": radam,
    "novograd": novograd_wrapper,
    "nadam": nadam,
    "ademamix": ademamix_wrapper,
    "lion": lion_wrapper,
    "adopt": adopt_wrapper,
}


def get_optimizer(params):
    """
    Returns an instance of the specified optimizer from the PyTorch `torch.optim` module.

    Args:
        params (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        optimizer (torch.optim.Optimizer): An instance of the specified optimizer.

    """
    # Retrieve the optimizer type from the input parameters
    optimizer_type = params["optimizer"]["type"]

    assert (
        optimizer_type in global_optimizer_dict
    ), f"Optimizer type {optimizer_type} not found"

    # Create the optimizer instance using the specified type and input parameters
    optimizer_function = global_optimizer_dict[optimizer_type]
    return optimizer_function(params)
