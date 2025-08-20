from torch.optim.lr_scheduler import (
    LambdaLR,
    CyclicLR,
    ExponentialLR,
    StepLR,
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
)
import math


def cyclical_lr(stepsize, min_lr, max_lr):
    # Scaler : we can adapt this if we do not want the triangular LR
    scaler = lambda x: 1
    # Lambda function to calculate the LR
    lr_lambda = lambda it: max_lr - (max_lr - min_lr) * relative(it, stepsize)

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
    mult = lambda it: max_lr_multiplier * rel_dist(
        it, cycle_length
    ) + min_lr_multiplier * (1 - rel_dist(it, cycle_length))

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
    parameters["scheduler"]["min_lr"] = parameters["scheduler"].get("min_lr", 10**-3)
    parameters["scheduler"]["max_lr"] = parameters["scheduler"].get("max_lr", 1)

    clr = cyclical_lr(
        parameters["scheduler"]["step_size"],
        min_lr=parameters["scheduler"]["min_lr"],
        max_lr=parameters["scheduler"]["max_lr"],
    )
    return LambdaLR(parameters["optimizer_object"], [clr])


def triangle_modified(parameters):
    # pick defaults
    parameters["scheduler"]["min_lr"] = parameters["scheduler"].get("min_lr", 0.000001)
    parameters["scheduler"]["max_lr"] = parameters["scheduler"].get("max_lr", 0.001)
    parameters["scheduler"]["max_lr_multiplier"] = parameters["scheduler"].get(
        "max_lr_multiplier", 1.0
    )

    clr = cyclical_lr_modified(
        parameters["scheduler"]["step_size"],
        parameters["scheduler"]["min_lr"],
        parameters["scheduler"]["max_lr"],
        parameters["scheduler"]["max_lr_multiplier"],
    )
    return LambdaLR(parameters["optimizer_object"], [clr])


def cyclic_lr_base(parameters, mode="triangular"):
    # pick defaults for "min_lr", "max_lr", "max_lr_multiplier" if not present in parameters
    parameters["scheduler"]["min_lr"] = parameters["scheduler"].get(
        "min_lr", parameters["learning_rate"] * 0.001
    )
    parameters["scheduler"]["max_lr"] = parameters["scheduler"].get(
        "max_lr", parameters["learning_rate"]
    )
    parameters["scheduler"]["gamma"] = parameters["scheduler"].get("gamma", 0.1)
    parameters["scheduler"]["scale_mode"] = parameters["scheduler"].get(
        "scale_mode", "cycle"
    )
    parameters["scheduler"]["cycle_momentum"] = parameters["scheduler"].get(
        "cycle_momentum", False
    )
    parameters["scheduler"]["base_momentum"] = parameters["scheduler"].get(
        "base_momentum", 0.8
    )
    parameters["scheduler"]["max_momentum"] = parameters["scheduler"].get(
        "max_momentum", 0.9
    )

    return CyclicLR(
        parameters["optimizer_object"],
        parameters["learning_rate"] * 0.001,  # min lr
        parameters["learning_rate"],  # mar_lr
        step_size_up=parameters["scheduler"]["step_size"],
        step_size_down=None,
        mode=mode,
        gamma=parameters["scheduler"]["gamma"],
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
    parameters["scheduler"]["gamma"] = parameters["scheduler"].get("gamma", 0.1)
    return ExponentialLR(
        parameters["optimizer_object"], parameters["scheduler"]["gamma"]
    )


def step(parameters):
    parameters["scheduler"]["gamma"] = parameters["scheduler"].get("gamma", 0.1)
    return StepLR(
        parameters["optimizer_object"],
        parameters["scheduler"]["step_size"],
        gamma=parameters["scheduler"]["gamma"],
    )


def reduce_on_plateau(parameters):
    parameters["scheduler"]["min_lr"] = parameters["scheduler"].get(
        "min_lr", parameters["learning_rate"] * 0.001
    )
    parameters["scheduler"]["gamma"] = parameters["scheduler"].get("gamma", 0.1)
    parameters["scheduler"]["mode"] = parameters["scheduler"].get("mde", "min")
    parameters["scheduler"]["threshold_mode"] = parameters["scheduler"].get(
        "threshold_mode", "rel"
    )
    parameters["scheduler"]["factor"] = parameters["scheduler"].get("factor", 0.1)
    parameters["scheduler"]["patience"] = parameters["scheduler"].get("patience", 10)
    parameters["scheduler"]["threshold"] = parameters["scheduler"].get(
        "threshold", 0.0001
    )
    parameters["scheduler"]["cooldown"] = parameters["scheduler"].get("cooldown", 0)

    return ReduceLROnPlateau(
        parameters["optimizer_object"],
        mode=parameters["scheduler"]["mode"],
        factor=parameters["scheduler"]["factor"],
        patience=parameters["scheduler"]["patience"],
        threshold=parameters["scheduler"]["threshold"],
        threshold_mode=parameters["scheduler"]["threshold_mode"],
        cooldown=parameters["scheduler"]["cooldown"],
        min_lr=parameters["scheduler"]["min_lr"],
    )


def cosineannealingwarmrestarts(parameters):
    parameters["scheduler"]["T_0"] = parameters["scheduler"].get("T_0", 5)
    parameters["scheduler"]["T_mult"] = parameters["scheduler"].get("T_mult", 1)
    parameters["scheduler"]["eta_min"] = parameters["scheduler"].get("eta_min", 0.001)

    return CosineAnnealingWarmRestarts(
        parameters["optimizer_object"],
        T_0=parameters["scheduler"]["T_0"],
        T_mult=parameters["scheduler"]["T_mult"],
        eta_min=parameters["scheduler"]["eta_min"],
    )


def cosineannealingLR(parameters):
    parameters["scheduler"]["T_max"] = parameters["scheduler"].get("T_max", 50)
    parameters["scheduler"]["eta_min"] = parameters["scheduler"].get("eta_min", 0.001)

    return CosineAnnealingLR(
        parameters["optimizer_object"],
        T_max=parameters["scheduler"]["T_max"],
        eta_min=parameters["scheduler"]["eta_min"],
    )
