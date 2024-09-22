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

from .ademamix import ademamix_wrapper

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

    # Create the optimizer instance using the specified type and input parameters
    if optimizer_type in global_optimizer_dict:
        optimizer_function = global_optimizer_dict[optimizer_type]
        return optimizer_function(params)
    else:
        raise ValueError("Optimizer type %s not found" % optimizer_type)
