from GANDLF.optimizers.wrap_torch import (
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
    "radam": radam,
}


def get_optimizers_gan(params):
    """
    Returns an instances of the specified optimizer from the PyTorch `torch.optim` module
    for both the generator and discriminator.
    Args:
        params (dict): A dictionary containing the input parameters for the optimizer.
    Returns:
        optimizer_gen (torch.optim.Optimizer): An instance of the specified optimizer for generator.
        optimizer_disc (torch.optim.Optimizer): An instance of the specified optimizer for discriminator.
    """
    optimizer_gen_type = params["optimizer_gen"]["type"]
    optimizer_disc_type = params["optimizer_disc"]["type"]
    if optimizer_gen_type in global_optimizer_dict:
        params["model_parameters"] = params["model_parameters_gen"]
        optimizer_gen_function = global_optimizer_dict[optimizer_gen_type]
        optimizer_gen = optimizer_gen_function(params)
    else:
        raise ValueError(
            "Generator optimizer type %s not found" % optimizer_gen_type
        )
    if optimizer_disc_type in global_optimizer_dict:
        params["model_parameters"] = params["model_parameters_disc"]
        optimizer_disc_function = global_optimizer_dict[optimizer_disc_type]
        optimizer_disc = optimizer_disc_function(params)
    else:
        raise ValueError(
            "Discriminator optimizer type %s not found" % optimizer_disc_type
        )

    return optimizer_gen, optimizer_disc
