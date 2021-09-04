from torch.optim.lr_scheduler import LambdaLR
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


def cyclical_lr_modified(cycle_length, min_lr=0.000001, max_lr=0.001):
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
    step_size = 4 * parameters["batch_size"] * parameters["training_samples_size"] # the latter needs to be added to the parameters in the training_loop, along with the optimizer
    # pick defaults for "min_lr", "max_lr" if not present in parameters
    # this should probably happen in parseConfig
    if not("min_lr" in parameters["scheduler"]):
        parameters["scheduler"]["min_lr"] = 10 ** -3
    if not("max_lr" in parameters["scheduler"]):
        parameters["scheduler"]["max_lr"] = 1
    clr = cyclical_lr(step_size, min_lr=parameters["scheduler"]["min_lr"], max_lr=parameters["scheduler"]["max_lr"])
    scheduler_lr = LambdaLR(parameters["optimizer"], [clr])

    return scheduler_lr