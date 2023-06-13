import sys
import torch
from torchmetrics import (
    StructuralSimilarityIndexMeasure,
    MeanSquaredError,
    MeanSquaredLogError,
    MeanAbsoluteError,
)
from GANDLF.utils import get_tensor_from_image


def structural_similarity_index(target, prediction, mask=None):
    """
    Computes the structural similarity index between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
        mask (torch.Tensor, optional): The mask tensor. Defaults to None.

    Returns:
        torch.Tensor: The structural similarity index.
    """
    ssim = StructuralSimilarityIndexMeasure(return_full_image=True)
    _, ssim_idx_full_image = ssim(target, prediction)
    mask = torch.ones_like(ssim_idx_full_image) if mask is None else mask
    try:
        ssim_idx = ssim_idx_full_image[mask]
    except Exception as e:
        print(f"Error: {e}")
        if len(ssim_idx_full_image.shape) == 0:
            ssim_idx = torch.ones_like(mask) * ssim_idx_full_image
    return ssim_idx.mean()


def mean_squared_error(target, prediction):
    """
    Computes the mean squared error between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
    """
    mse = MeanSquaredError()
    return mse(target, prediction)


def peak_signal_noise_ratio(target, prediction, mask=None):
    """
    Computes the peak signal to noise ratio between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
        mask (torch.Tensor, optional): The mask tensor. Defaults to None.
    """
    mse = mean_squared_error(target, prediction, mask)
    return (
        10.0
        * torch.log10((torch.max(target) - torch.min(target)) ** 2)
        / (mse + sys.float_info.epsilon)
    )


def mean_squared_log_error(target, prediction):
    """
    Computes the mean squared log error between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
    """
    mle = MeanSquaredLogError()
    return mle(target, prediction)


def mean_absolute_error(target, prediction):
    """
    Computes the mean absolute error between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
    """
    mae = MeanAbsoluteError()
    return mae(target, prediction)
