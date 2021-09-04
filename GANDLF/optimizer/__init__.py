from .wrap_torch import sgd, asgd, adam, rprop

global_optimizer_dict = {
    'sgd': sgd,
    'asgd': asgd,
    'adam': adam,
    'rprop': rprop,
}