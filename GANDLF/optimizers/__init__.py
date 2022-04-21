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
)

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
}


def get_optimizer(params):
    """
    Function to get the optimizer definition.

    Args:
        params (dict): The parameters' dictionary.

    Returns:
        model (Optimizer): The optimizer definition.
    """
    return global_optimizer_dict[params["optimizer"]["type"]](params)
