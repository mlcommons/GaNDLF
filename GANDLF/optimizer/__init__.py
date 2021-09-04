from .wrap_torch import sgd, asgd, adam, adamw, adamax, sparseadam, rprop, adadelta

global_optimizer_dict = {
    "sgd": sgd,
    "asgd": asgd,
    "adam": adam,
    "adamw": adamw,
    "adamax": adamax,
    "sparseadam": sparseadam,
    "rprop": rprop,
    "adadelta": adadelta,
}