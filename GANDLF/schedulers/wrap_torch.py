import math

from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    ReduceLROnPlateau,
    StepLR,
)


def cyclical_lr(stepsize, min_lr, max_lr):
    # Scaler : we can adapt this if we do not want the triangular LR
    def scaler(x):
        return 1

    # Lambda function to calculate the LR
    def lr_lambda(it):
        return max_lr - (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def cyclical_lr_modified(cycle_length, min_lr, max_lr, max_lr_multiplier):
    min_lr_multiplier = min_lr / max_lr
    max_lr_multiplier = 1.0

    # Lambda function to calculate what to multiply the initial learning rate by
    # The beginning and end of the cycle result in highest multipliers (lowest at the center)
    def mult(it):
        return max_lr_multiplier * rel_dist(it, cycle_length) + min_lr_multiplier * (
            1 - rel_dist(it, cycle_length)
        )

    def rel_dist(iteration, cycle_length):
        # relative_distance from iteration to the center of the cycle
        # equal to 1 at beggining of cycle and 0 right at the cycle center

        # reduce the iteration to less than the cycle length
        iteration = iteration % cycle_length
        return 2 * abs(iteration - cycle_length / 2.0) / cycle_length

    return mult


def base_triangle(parameters):
    """
    This function parses the parameters from the config file and returns the appropriate object
    """

    # pick defaults
    if "min_lr" not in parameters["scheduler"]:
        parameters["scheduler"]["min_lr"] = 10**-3
    if "max_lr" not in parameters["scheduler"]:
        parameters["scheduler"]["max_lr"] = 1

    clr = cyclical_lr(
        parameters["scheduler"]["step_size"],
        min_lr=parameters["scheduler"]["min_lr"],
        max_lr=parameters["scheduler"]["max_lr"],
    )
    return LambdaLR(parameters["optimizer_object"], [clr])


def triangle_modified(parameters):
    # pick defaults
    if "min_lr" not in parameters["scheduler"]:
        parameters["scheduler"]["min_lr"] = 0.000001
    if "max_lr" not in parameters["scheduler"]:
        parameters["scheduler"]["max_lr"] = 0.001
    if "max_lr_multiplier" not in parameters["scheduler"]:
        parameters["scheduler"]["max_lr_multiplier"] = 1.0

    clr = cyclical_lr_modified(
        parameters["scheduler"]["step_size"],
        parameters["scheduler"]["min_lr"],
        parameters["scheduler"]["max_lr"],
        parameters["scheduler"]["max_lr_multiplier"],
    )
    return LambdaLR(parameters["optimizer_object"], [clr])


def cyclic_lr_base(parameters, mode="triangular"):
    # pick defaults for "min_lr", "max_lr", "max_lr_multiplier" if not present in parameters
    if "min_lr" not in parameters["scheduler"]:
        parameters["scheduler"]["min_lr"] = parameters["learning_rate"] * 0.001
    if "max_lr" not in parameters["scheduler"]:
        parameters["scheduler"]["max_lr"] = parameters["learning_rate"]
    if "gamma" not in parameters["scheduler"]:
        parameters["scheduler"]["gamma"] = 0.1
    if "scale_mode" not in parameters["scheduler"]:
        parameters["scheduler"]["scale_mode"] = "cycle"
    if "cycle_momentum" not in parameters["scheduler"]:
        parameters["scheduler"]["cycle_momentum"] = False
    if "base_momentum" not in parameters["scheduler"]:
        parameters["scheduler"]["base_momentum"] = 0.8
    if "max_momentum" not in parameters["scheduler"]:
        parameters["scheduler"]["max_momentum"] = 0.9

    return CyclicLR(
        parameters["optimizer_object"],
        parameters["learning_rate"] * 0.001,
        parameters["learning_rate"],
        step_size_up=parameters["scheduler"]["step_size"],
        step_size_down=None,
        mode=mode,
        gamma=1.0,
        scale_fn=None,
        scale_mode=parameters["scheduler"]["scale_mode"],
        cycle_momentum=parameters["scheduler"]["cycle_momentum"],
        base_momentum=parameters["scheduler"]["base_momentum"],
        max_momentum=parameters["scheduler"]["max_momentum"],
    )


## this is not working with default step_size, for some reason
# def cyclic_lr_triangular2(parameters):
#     return cyclic_lr_base(parameters, mode="triangular2")


def cyclic_lr_exp_range(parameters):
    return cyclic_lr_base(parameters, mode="exp_range")


def exp(parameters):
    if "gamma" not in parameters["scheduler"]:
        parameters["scheduler"]["gamma"] = 0.1
    return ExponentialLR(
        parameters["optimizer_object"], parameters["scheduler"]["gamma"]
    )


def step(parameters):
    if "gamma" not in parameters["scheduler"]:
        parameters["scheduler"]["gamma"] = 0.1
    return StepLR(
        parameters["optimizer_object"],
        parameters["scheduler"]["step_size"],
        gamma=parameters["scheduler"]["gamma"],
    )


def reduce_on_plateau(parameters):
    if "min_lr" not in parameters["scheduler"]:
        parameters["scheduler"]["min_lr"] = parameters["learning_rate"] * 0.001
    if "gamma" not in parameters["scheduler"]:
        parameters["scheduler"]["gamma"] = 0.1
    if "mode" not in parameters["scheduler"]:
        parameters["scheduler"]["mde"] = "min"
    if "threshold_mode" not in parameters["scheduler"]:
        parameters["scheduler"]["threshold_mode"] = "rel"
    if "factor" not in parameters["scheduler"]:
        parameters["scheduler"]["factor"] = 0.1
    if "patience" not in parameters["scheduler"]:
        parameters["scheduler"]["patience"] = 10
    if "threshold" not in parameters["scheduler"]:
        parameters["scheduler"]["threshold"] = 0.0001
    if "cooldown" not in parameters["scheduler"]:
        parameters["scheduler"]["cooldown"] = 0

    return ReduceLROnPlateau(
        parameters["optimizer_object"],
        mode=parameters["scheduler"]["mde"],
        factor=parameters["scheduler"]["factor"],
        patience=parameters["scheduler"]["patience"],
        threshold=parameters["scheduler"]["threshold"],
        threshold_mode=parameters["scheduler"]["threshold_mode"],
        cooldown=parameters["scheduler"]["cooldown"],
        min_lr=parameters["scheduler"]["min_lr"],
    )


def cosineannealing(parameters):
    if "T_0" not in parameters["scheduler"]:
        parameters["scheduler"]["T_0"] = 5
    if "T_mult" not in parameters["scheduler"]:
        parameters["scheduler"]["T_mult"] = 1
    if "min_lr" not in parameters["scheduler"]:
        parameters["scheduler"]["min_lr"] = parameters["learning_rate"] * 0.001

    return CosineAnnealingWarmRestarts(
        parameters["optimizer_object"],
        T_0=parameters["scheduler"]["T_0"],
        T_mult=parameters["scheduler"]["T_mult"],
        eta_min=parameters["scheduler"]["min_lr"],
    )
