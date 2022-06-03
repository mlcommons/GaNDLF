from .wrap_torch import (
    adadelta,
    adagrad,
    adam,
    adamax,
    adamw,
    asgd,
    rmsprop,
    rprop,
    sgd,
)

# from .wrap_torch import sparseadam

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
