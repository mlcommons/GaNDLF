from .wrap_torch import sgd, asgd, adam, adamw, adamax, rprop

global_optimizer_dict = {
    "sgd": sgd,
    "asgd": asgd,
    "adam": adam,
    "adamw": adamw,
    "adamax": adamax,
    "rprop": rprop,
}