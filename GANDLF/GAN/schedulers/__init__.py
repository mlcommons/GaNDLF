from GANDLF.schedulers.wrap_torch import (
    base_triangle,
    triangle_modified,
    cyclic_lr_base,
    cyclic_lr_exp_range,
    exp,
    step,
    reduce_on_plateau,
    cosineannealing,
)


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
    "cosineannealing": cosineannealing,
}


def get_scheduler_gan(params):
    """
    Function to get the scheduler definition for both the generator and discriminator.

    Args:
        params (dict): The parameters' dictionary.

    Returns:
        model (object): The scheduler definition.
    """
    scheduler_gen_type = params["scheduler_g"]["type"]
    scheduler_disc_type = params["scheduler_d"]["type"]
    if scheduler_gen_type in global_schedulers_dict:
        scheduler_gen_function = global_schedulers_dict[scheduler_gen_type]
        # a trick for compatibility with wrap_torch.py wrappers
        params["scheduler"] = params["scheduler_g"]
        params["optimizer_object"] = params["optimizer_gen_object"]
        scheduler_gen = scheduler_gen_function(params)
        del params["scheduler"]
        del params["optimizer_object"]
    else:
        raise ValueError(
            "Genertor scheduler type %s not found" % scheduler_gen_type
        )
    if scheduler_disc_type in global_schedulers_dict:
        scheduler_disc_function = global_schedulers_dict[scheduler_disc_type]
        # a trick for compatibility with wrap_torch.py wrappers
        params["scheduler"] = params["scheduler_d"]
        params["optimizer_object"] = params["optimizer_disc_object"]
        scheduler_disc = scheduler_disc_function(params)
        del params["scheduler"]
        del params["optimizer_object"]
    else:
        raise ValueError(
            "Discriminator scheduler type %s not found" % scheduler_disc_type
        )
    return scheduler_gen, scheduler_disc
