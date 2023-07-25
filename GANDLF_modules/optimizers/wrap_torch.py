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
    RAdam,
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
        momentum=parameters["optimizer"].get("momentum", 0.9),
        weight_decay=parameters["optimizer"].get("weight_decay", 0),
        dampening=parameters["optimizer"].get("dampening", 0),
        nesterov=parameters["optimizer"].get("Nesterov", False),
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
        alpha=parameters["optimizer"].get("alpha", 0.75),
        t0=parameters["optimizer"].get("t0", 1e6),
        lambd=parameters["optimizer"].get("lambd", 1e-4),
        weight_decay=parameters["optimizer"].get("weight_decay", 0),
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
        betas=parameters["optimizer"].get("betas", (0.9, 0.999)),
        weight_decay=parameters["optimizer"].get("weight_decay", 0.00005),
        eps=parameters["optimizer"].get("eps", 1e-8),
        amsgrad=parameters["optimizer"].get("amsgrad", False),
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
        betas=parameters["optimizer"].get("betas", (0.9, 0.999)),
        weight_decay=parameters["optimizer"].get("weight_decay", 0.00005),
        eps=parameters["optimizer"].get("eps", 1e-8),
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
        etas=parameters["optimizer"].get("etas", (0.5, 1.2)),
        step_sizes=parameters["optimizer"].get("step_sizes", (1e-7, 50)),
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
        rho=parameters["optimizer"].get("rho", 0.9),
        eps=parameters["optimizer"].get("eps", 1e-6),
        weight_decay=parameters["optimizer"].get("weight_decay", 0),
    )


def adagrad(parameters):
    """
    Creates an Adagrad optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        parameters (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        optimizer (torch.optim.Adagrad): An Adagrad optimizer.

    """

    # Create the optimizer using the input parameters
    return Adagrad(
        parameters["model_parameters"],
        lr=parameters.get("learning_rate"),
        lr_decay=parameters["optimizer"].get("lr_decay", 0),
        eps=parameters["optimizer"].get("eps", 1e-6),
        weight_decay=parameters["optimizer"].get("weight_decay", 0),
    )


def rmsprop(parameters):
    """
    Creates an RMSprop optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        parameters (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        optimizer (torch.optim.RMSprop): An RMSprop optimizer.

    """
    # Create the optimizer using the input parameters
    return RMSprop(
        parameters["model_parameters"],
        lr=parameters.get("learning_rate"),
        alpha=parameters["optimizer"].get("alpha", 0.99),
        eps=parameters["optimizer"].get("eps", 1e-8),
        centered=parameters["optimizer"].get("centered", False),
        momentum=parameters["optimizer"].get("momentum", 0),
        weight_decay=parameters["optimizer"].get("weight_decay", 0),
    )


def radam(parameters):
    """
    Creates a RAdam optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        parameters (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        optimizer (torch.optim.RAdam): A RAdam optimizer.
    """
    # Create the optimizer using the input parameters
    return RAdam(
        parameters["model_parameters"],
        lr=parameters.get("learning_rate"),
        betas=parameters["optimizer"].get("betas", (0.9, 0.999)),
        eps=parameters["optimizer"].get("eps", 1e-8),
        weight_decay=parameters["optimizer"].get("weight_decay", 0),
        foreach=parameters["optimizer"].get("foreach", None),
    )
