from torch.optim import (
    SGD,
    ASGD,
    Rprop,
    Adam,
    AdamW,
    # SparseAdam,
    Adamax,
    Adadelta,
    Adagrad,
    RMSprop,
)


def sgd(parameters):
    """
    Creates a Stochastic Gradient Descent optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        parameters (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        optimizer (torch.optim.SGD): A Stochastic Gradient Descent optimizer.

    """
    # Create the optimizer using the input parameters
    optimizer = SGD(
        parameters["model_parameters"],
        lr=parameters.get("learning_rate"),
        momentum=optimizer_params.get("momentum", 0.9),
        weight_decay=optimizer_params.get("weight_decay", 0),
        dampening=optimizer_params.get("dampening", 0),
        Nesterov=optimizer_params.get("nesterov", False),
    )

    return optimizer


def asgd(parameters):
    """
    Creates an Averaged Stochastic Gradient Descent optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        parameters (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        optimizer (torch.optim.ASGD): An Averaged Stochastic Gradient Descent optimizer.

    """
    # Create the optimizer using the input parameters
    return ASGD(
        parameters["model_parameters"],
        lr=parameters.get("learning_rate"),
        alpha=optimizer_params.get("alpha", 0.75),
        t0=optimizer_params.get("t0", 1e6),
        lambd=optimizer_params.get("lambd", 1e-4),
        weight_decay=optimizer_params.get("weight_decay", 0),
    )


def adam(parameters, opt_type="normal"):
    """
    Creates an Adam or AdamW optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        parameters (dict): A dictionary containing the input parameters for the optimizer.
        opt_type (str): A string indicating the type of optimizer to create (either "normal" for Adam or "AdamW" for AdamW).

    Returns:
        optimizer (torch.optim.Adam or torch.optim.AdamW): An Adam or AdamW optimizer.

    """
    # Determine which optimizer to create based on opt_type
    if opt_type == "normal":
        optimizer_fn = Adam
    elif opt_type == "AdamW":
        optimizer_fn = AdamW
    else:
        raise ValueError(f"Invalid optimizer type: {opt_type}")

    # Create the optimizer using the input parameters
    return optimizer_fn(
        parameters["model_parameters"],
        lr=parameters.get("learning_rate"),
        betas=optimizer_params.get("betas", (0.9, 0.999)),
        weight_decay=optimizer_params.get("weight_decay", 0.00005),
        eps=optimizer_params.get("eps", 1e-8),
        amsgrad=optimizer_params.get("amsgrad", False),
    )


def adamw(parameters):
    """
    Creates an AdamW optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        parameters (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        optimizer (torch.optim.AdamW): An AdamW optimizer.

    """
    return adam(parameters, opt_type="AdamW")

def adamax(parameters):
    """
    Creates an Adamax optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        parameters (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        optimizer (torch.optim.Adamax): An Adamax optimizer.

    """
    # Create the optimizer using the input parameters
    return Adamax(
        parameters["model_parameters"],
        lr=parameters.get("learning_rate"),
        betas=optimizer_params.get("betas", (0.9, 0.999)),
        weight_decay=optimizer_params.get("weight_decay", 0.00005),
        eps=optimizer_params.get("eps", 1e-8),
    )


# def sparseadam(parameters):
#     # pick defaults
#     if not ("betas" in parameters["optimizer"]):
#         parameters["optimizer"]["betas"] = (0.9, 0.999)
#     if not ("eps" in parameters["optimizer"]):
#         parameters["optimizer"]["eps"] = 1e-8

#     return SparseAdam(
#         parameters["model_parameters"],
#         lr=parameters["learning_rate"],
#         betas=parameters["optimizer"]["betas"],
#         eps=parameters["optimizer"]["eps"],
#     )


def rprop(parameters):
    """
    Creates a Resilient Backpropagation optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        parameters (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        optimizer (torch.optim.Rprop): A Resilient Backpropagation optimizer.

    """
    # Create the optimizer using the input parameters
    return Rprop(
        parameters["model_parameters"],
        lr=parameters.get("learning_rate"),
        etas=optimizer_params.get("etas", (0.5, 1.2)),
        step_sizes=optimizer_params.get("step_sizes", (1e-7, 50)),
    )


def adadelta(parameters):
    """
    Creates an Adadelta optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        parameters (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        optimizer (torch.optim.Adadelta): An Adadelta optimizer.

    """
    # Create the optimizer using the input parameters
    return Adadelta(
        parameters["model_parameters"],
        lr=parameters.get("learning_rate"),
        rho=optimizer_params.get("rho", 0.9),
        eps=optimizer_params.get("eps", 1e-6),
        weight_decay=optimizer_params.get("weight_decay", 0),
    )


def adagrad(parameters):
    """
    Creates an Adagrad optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        parameters (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        optimizer (torch.optim.Adagrad): An Adagrad optimizer.

    """
    # Set default values for optimizer parameters
    optimizer_params = parameters.get("optimizer", {})
    lr_decay = optimizer_params.get("lr_decay", 0)
    eps = optimizer_params.get("eps", 1e-6)
    weight_decay = optimizer_params.get("weight_decay", 0)

    # Create the optimizer using the input parameters
    return Adagrad(
        parameters["model_parameters"],
        lr=parameters.get("learning_rate"),
        lr_decay=optimizer_params.get("lr_decay", 0),
        eps=optimizer_params.get("eps", 1e-6),
        weight_decay=optimizer_params.get("weight_decay", 0),
    )


def rmsprop(parameters):
    """
    Creates an RMSprop optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        parameters (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        optimizer (torch.optim.RMSprop): An RMSprop optimizer.

    """
    # Set default values for optimizer parameters
    optimizer_params = parameters.get("optimizer", {})
    momentum = optimizer_params.get("momentum", 0)
    weight_decay = optimizer_params.get("weight_decay", 0)
    alpha = optimizer_params.get("alpha", 0.99)
    eps = optimizer_params.get("eps", 1e-8)
    centered = optimizer_params.get("centered", False)

    # Create the optimizer using the input parameters
    return RMSprop(
        parameters["model_parameters"],
        lr=parameters.get("learning_rate"),
        alpha=alpha,
        eps=eps,
        centered=centered,
        momentum=momentum,
        weight_decay=weight_decay,
    )

