# -*- coding: utf-8 -*-
"""
Implementation of Adaptive gradient clipping
"""
from typing import List, Optional
import torch


def unitwise_norm(x: torch.Tensor, norm_type: Optional[float] = 2.0) -> torch.Tensor:
    """
    Compute norms of each weight tensor in a model, and return the global norm.

    Args:
        x (torch.Tensor): The input tensor.
        norm_type (Optional[float], optional): The type of norm to compute. Defaults to 2.0.

    Returns:
        torch.Tensor: The global norm.
    """
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        # This should work for nn.ConvNd (where N is the number of dimensions)
        # and nn.Linear where number of output dimensions is first
        # in the weight/kernel tensor passed.
        # This might need something for a few unplanned scenarios
        # but we dont have to deal with them here.
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)


def adaptive_gradient_clip_(
    parameters: List[torch.Tensor],
    clip_factor: Optional[float] = 0.01,
    eps: Optional[float] = 1e-3,
    norm_type: Optional[float] = 2.0,
) -> None:
    """
    Performs adaptive gradient clipping on the parameters of a PyTorch model.

    Args:
        parameters (List[torch.Tensor]): The parameters to be clipped.
        clip_factor (Optional[float], optional): The clipping factor. Defaults to 0.01.
        eps (Optional[float], optional): The epsilon value. Defaults to 1e-3.
        norm_type (Optional[float], optional): The type of norm to compute. Defaults to 2.0.

    Adaptive Gradient Clipping
    Original implementation of Adaptive Gradient Clipping derived from
    An impl of AGC, as per (https://arxiv.org/abs/2102.06171):


    Paper Name : High-Performance Large-Scale Image Recognition Without Normalization
    Authors : Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan
    Published : arXiv preprint arXiv: 2021

    Code references:
      * Official JAX impl (paper authors): https://github.com/deepmind/deepmind-research/tree/master/nfnets
      * Phil Wang's PyTorch gist: https://gist.github.com/lucidrains/0d6560077edac419ab5d3aa29e674d5c
    """
    # If parameter is not a list, make it one so that you can iterate over it
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Parse through the list of parameters
    for p in parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        max_norm = (
            unitwise_norm(p_data, norm_type=norm_type).clamp_(min=eps).mul_(clip_factor)
        )
        grad_norm = unitwise_norm(g_data, norm_type=norm_type)
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grads)
