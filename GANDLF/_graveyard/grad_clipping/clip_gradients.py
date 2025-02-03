# -*- coding: utf-8 -*-
"""
Implementation of functions to clip gradients
"""
from typing import Optional
import torch
from GANDLF.grad_clipping.adaptive_gradient_clipping import adaptive_gradient_clip_


def dispatch_clip_grad_(
    parameters: torch.Tensor,
    value: float,
    mode: Optional[str] = "norm",
    norm_type: Optional[float] = 2.0,
) -> None:
    """
    Dispatches the gradient clipping method to the corresponding function based on the mode.

    Args:
        parameters (torch.Tensor): The model parameters to be clipped.
        value (float): The clipping value/factor/norm, mode dependent.
        mode (Optional[str], optional): The mode of clipping. Defaults to "norm".
        norm_type (Optional[float], optional): The type of norm to compute. Defaults to 2.0.
    """
    assert mode in ["norm", "value", "agc"], f"Unknown clip mode ({mode})."
    if mode == "norm":
        torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
    elif mode == "value":
        torch.nn.utils.clip_grad_value_(parameters, value)
    elif mode == "agc":
        adaptive_gradient_clip_(parameters, value, norm_type=norm_type)
