from .wrap_torch import sgd, asgd

global_optimizer_dict = {
    'sgd': sgd,
    'asgd': asgd,
}