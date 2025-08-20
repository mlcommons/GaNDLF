from .wrap_torch import (
    base_triangle,
    triangle_modified,
    cyclic_lr_base,
    cyclic_lr_exp_range,
    exp,
    step,
    reduce_on_plateau,
    cosineannealingwarmrestarts,
    cosineannealingLR,
)

from .wrap_monai import warmupcosineschedule

# defining dict for schedulers - key is the string and the value is the transform object
global_schedulers_dict = {
    "triangle": base_triangle,
    "triangle_modified": triangle_modified,
    "triangular": cyclic_lr_base,
    "exp_range": cyclic_lr_exp_range,
    "exp": exp,
    "exponential": exp,
    "step": step,
    "reduce_on_plateau": reduce_on_plateau,
    "reduce-on-plateau": reduce_on_plateau,
    "plateau": reduce_on_plateau,
    "reduceonplateau": reduce_on_plateau,
    "cosineannealing": cosineannealingwarmrestarts,
    "cosineannealingwarmrestarts": cosineannealingwarmrestarts,
    "cosineannealinglr": cosineannealingLR,
    "warmupcosineschedule": warmupcosineschedule,
    "wcs": warmupcosineschedule,
}


def get_scheduler(params):
    """
    Function to get the scheduler definition.

    Args:
        params (dict): The parameters' dictionary.

    Returns:
        scheduler (object): The scheduler definition.
    """
    chosen_scheduler = params["scheduler"]["type"].lower()
    assert (
        chosen_scheduler in global_schedulers_dict
    ), f"Could not find the requested scheduler '{params['scheduler']['type']}'"
    return global_schedulers_dict[chosen_scheduler](params)
