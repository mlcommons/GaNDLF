from torch.optim.lr_scheduler import LambdaLR, CyclicLR, ExponentialLR, StepLR
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
    if not ("min_lr" in parameters["scheduler"]):
        parameters["scheduler"]["min_lr"] = 10 ** -3
    if not ("max_lr" in parameters["scheduler"]):
        parameters["scheduler"]["max_lr"] = 1

    clr = cyclical_lr(
        parameters["scheduler"]["step_size"],
        min_lr=parameters["scheduler"]["min_lr"],
        max_lr=parameters["scheduler"]["max_lr"],
    )
    return LambdaLR(parameters["optimizer"], [clr])


def triangle_modified(parameters):

    # pick defaults
    if not ("min_lr" in parameters["scheduler"]):
        parameters["scheduler"]["min_lr"] = 0.000001
    if not ("max_lr" in parameters["scheduler"]):
        parameters["scheduler"]["max_lr"] = 0.001
    if not ("max_lr_multiplier" in parameters["scheduler"]):
        parameters["scheduler"]["max_lr_multiplier"] = 1.0

    clr = cyclical_lr_modified(
        parameters["scheduler"]["step_size"],
        parameters["scheduler"]["min_lr"],
        parameters["scheduler"]["max_lr"],
        parameters["scheduler"]["max_lr_multiplier"],
    )
    return LambdaLR(parameters["optimizer"], [clr])

def cyclic_lr_base(parameters, mode="triangular"):
    # pick defaults for "min_lr", "max_lr", "max_lr_multiplier" if not present in parameters
    if not ("min_lr" in parameters["scheduler"]):
        parameters["scheduler"]["min_lr"] = parameters["learning_rate"] * 0.001
    if not ("max_lr" in parameters["scheduler"]):
        parameters["scheduler"]["max_lr"] = parameters["learning_rate"]
    if not ("gamma" in parameters["scheduler"]):
        parameters["scheduler"]["gamma"] = 0.1
    if not ("scale_mode" in parameters["scheduler"]):
        parameters["scheduler"]["scale_mode"] = "cycle"
    if not ("cycle_momentum" in parameters["scheduler"]):
        parameters["scheduler"]["cycle_momentum"] = False
    if not ("base_momentum" in parameters["scheduler"]):
        parameters["scheduler"]["base_momentum"] = 0.8
    if not ("max_momentum" in parameters["scheduler"]):
        parameters["scheduler"]["max_momentum"] = 0.9
    
    return CyclicLR(
        parameters["optimizer"],
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

def cyclic_lr_triangular2(parameters):
    return cyclic_lr_base(parameters, mode="triangular2")

def cyclic_lr_exp_range(parameters):
    return cyclic_lr_base(parameters, mode="exp_range")

def exp(parameters):
    if not ("gamma" in parameters["scheduler"]):
        parameters["scheduler"]["gamma"] = 0.1
    return ExponentialLR(parameters["optimizer"], parameters["scheduler"]["gamma"])

def step(parameters):
    if not ("gamma" in parameters["scheduler"]):
        parameters["scheduler"]["gamma"] = 0.1
    return StepLR(parameters["optimizer"], parameters["step_size"], gamma=parameters["learning_rate"])