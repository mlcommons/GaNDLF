from torch.optim import (  # SparseAdam,
    ASGD,
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    RMSprop,
    Rprop,
)


def sgd(parameters):
    # pick defaults
    if not ("momentum" in parameters["optimizer"]):
        parameters["optimizer"]["momentum"] = 0.9
    if not ("weight_decay" in parameters["optimizer"]):
        parameters["optimizer"]["weight_decay"] = 0
    if not ("dampening" in parameters["optimizer"]):
        parameters["optimizer"]["dampening"] = 0
    if not ("nesterov" in parameters["optimizer"]):
        parameters["optimizer"]["nesterov"] = False
    return SGD(
        parameters["model_parameters"],
        lr=parameters["learning_rate"],
        momentum=parameters["optimizer"]["momentum"],
        weight_decay=parameters["optimizer"]["weight_decay"],
        dampening=parameters["optimizer"]["dampening"],
        nesterov=parameters["optimizer"]["nesterov"],
    )


def asgd(parameters):
    # pick defaults
    if not ("lambd" in parameters["optimizer"]):
        parameters["optimizer"]["lambd"] = 1e-4
    if not ("alpha" in parameters["optimizer"]):
        parameters["optimizer"]["alpha"] = 0.75
    if not ("t0" in parameters["optimizer"]):
        parameters["optimizer"]["t0"] = 1e6
    if not ("weight_decay" in parameters["optimizer"]):
        parameters["optimizer"]["weight_decay"] = 0
    return ASGD(
        parameters["model_parameters"],
        lr=parameters["learning_rate"],
        alpha=parameters["optimizer"]["alpha"],
        t0=parameters["optimizer"]["t0"],
        lambd=parameters["optimizer"]["lambd"],
        weight_decay=parameters["optimizer"]["weight_decay"],
    )


def adam(parameters, opt_type="normal"):
    # pick defaults
    if not ("betas" in parameters["optimizer"]):
        parameters["optimizer"]["betas"] = (0.9, 0.999)
    if not ("weight_decay" in parameters["optimizer"]):
        parameters["optimizer"]["weight_decay"] = 0.00005
    if not ("eps" in parameters["optimizer"]):
        parameters["optimizer"]["eps"] = 1e-8
    if not ("amsgrad" in parameters["optimizer"]):
        parameters["optimizer"]["amsgrad"] = False

    if opt_type == "normal":
        function = Adam
    else:
        function = AdamW
    return function(
        parameters["model_parameters"],
        lr=parameters["learning_rate"],
        betas=parameters["optimizer"]["betas"],
        weight_decay=parameters["optimizer"]["weight_decay"],
        eps=parameters["optimizer"]["eps"],
        amsgrad=parameters["optimizer"]["amsgrad"],
    )


def adamw(parameters):
    return adam(parameters, opt_type="adamw")


def adamax(parameters):
    # pick defaults
    if not ("betas" in parameters["optimizer"]):
        parameters["optimizer"]["betas"] = (0.9, 0.999)
    if not ("weight_decay" in parameters["optimizer"]):
        parameters["optimizer"]["weight_decay"] = 0.00005
    if not ("eps" in parameters["optimizer"]):
        parameters["optimizer"]["eps"] = 1e-8

    return Adamax(
        parameters["model_parameters"],
        lr=parameters["learning_rate"],
        betas=parameters["optimizer"]["betas"],
        weight_decay=parameters["optimizer"]["weight_decay"],
        eps=parameters["optimizer"]["eps"],
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
    if not ("etas" in parameters["optimizer"]):
        parameters["optimizer"]["etas"] = (0.5, 1.2)
    if not ("step_sizes" in parameters["optimizer"]):
        parameters["optimizer"]["step_sizes"] = (1e-7, 50)
    return Rprop(
        parameters["model_parameters"],
        lr=parameters["learning_rate"],
        etas=parameters["optimizer"]["etas"],
        step_sizes=parameters["optimizer"]["step_sizes"],
    )


def adadelta(parameters):
    # pick defaults
    if not ("rho" in parameters["optimizer"]):
        parameters["optimizer"]["rho"] = 0.9
    if not ("eps" in parameters["optimizer"]):
        parameters["optimizer"]["eps"] = 1e-6
    if not ("weight_decay" in parameters["optimizer"]):
        parameters["optimizer"]["weight_decay"] = 0
    return Adadelta(
        parameters["model_parameters"],
        lr=parameters["learning_rate"],
        rho=parameters["optimizer"]["rho"],
        eps=parameters["optimizer"]["eps"],
        weight_decay=parameters["optimizer"]["weight_decay"],
    )


def adagrad(parameters):
    # pick defaults
    if not ("lr_decay" in parameters["optimizer"]):
        parameters["optimizer"]["lr_decay"] = 0
    if not ("eps" in parameters["optimizer"]):
        parameters["optimizer"]["eps"] = 1e-6
    if not ("weight_decay" in parameters["optimizer"]):
        parameters["optimizer"]["weight_decay"] = 0

    return Adagrad(
        parameters["model_parameters"],
        lr=parameters["learning_rate"],
        lr_decay=parameters["optimizer"]["lr_decay"],
        eps=parameters["optimizer"]["eps"],
        weight_decay=parameters["optimizer"]["weight_decay"],
    )


def rmsprop(parameters):
    # pick defaults
    if not ("momentum" in parameters["optimizer"]):
        parameters["optimizer"]["momentum"] = 0
    if not ("weight_decay" in parameters["optimizer"]):
        parameters["optimizer"]["weight_decay"] = 0
    if not ("alpha" in parameters["optimizer"]):
        parameters["optimizer"]["alpha"] = 0.99
    if not ("eps" in parameters["optimizer"]):
        parameters["optimizer"]["eps"] = 1e-8
    if not ("centered" in parameters["optimizer"]):
        parameters["optimizer"]["centered"] = False

    return RMSprop(
        parameters["model_parameters"],
        lr=parameters["learning_rate"],
        alpha=parameters["optimizer"]["alpha"],
        eps=parameters["optimizer"]["eps"],
        centered=parameters["optimizer"]["centered"],
        momentum=parameters["optimizer"]["momentum"],
        weight_decay=parameters["optimizer"]["weight_decay"],
    )
