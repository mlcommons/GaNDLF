from .wrap_torch import base_triangle, triangle_modified, cyclic_lr_base, cyclic_lr_triangular2, cyclic_lr_exp_range, exp, step, reduce_on_plateau


# defining dict for schedulers - key is the string and the value is the transform object
global_schedulers_dict = {
    "triangle": base_triangle,
    "triangle_modified": triangle_modified,
    "triangular": cyclic_lr_base,
    "triangular2": cyclic_lr_triangular2,
    "exp_range": cyclic_lr_exp_range,
    "exp": exp,
    "step": step,
    "reduce_on_plateau": reduce_on_plateau,
    "reduce-on-plateau": reduce_on_plateau,
    "plateau": reduce_on_plateau,
    "reduceonplateau": reduce_on_plateau,
}
