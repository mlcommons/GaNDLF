# -*- coding: utf-8 -*-
"""
Implementation of functions to clip gradients
"""

import torch
from GANDLF.grad_clipping.adaptive_gradient_clipping import adaptive_gradient_clip_


def dispatch_clip_grad_(
    parameters, value: float, mode: str = "norm", norm_type: float = 2.0
):
    """
    Dispatches the gradient clipping method to the corresponding function based on the mode.

    Args:
        parameters (Iterable): The model parameters to be clipped.
        value (float): The clipping value/factor/norm, mode dependent.
        mode (str): The clipping mode, one of 'norm', 'value', 'agc' (default: 'norm').
        norm_type (float): The p-norm to use for computing the norm of the gradients (default: 2.0).
    """
    if mode == "norm":
        torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
    elif mode == "value":
        torch.nn.utils.clip_grad_value_(parameters, value)
    elif mode == "agc":
        adaptive_gradient_clip_(parameters, value, norm_type=norm_type)
    else:
        assert False, f"Unknown clip mode ({mode})."
