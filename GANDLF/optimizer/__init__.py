from .wrap_torch import sgd, asgd, adam

global_optimizer_dict = {
    'sgd': sgd,
    'asgd': asgd,
    'adam': adam,
}