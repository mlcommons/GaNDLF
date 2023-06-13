import torch
from torchmetrics import (
    StructuralSimilarityIndexMeasure,
    MeanSquaredError,
)


def structural_similarity_index(target, prediction, params, mask=None):
    """
    Computes the structural similarity index between the target and prediction.

    Args:
        target (torch.Tensor): The target tensor.
        prediction (torch.Tensor): The prediction tensor.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        torch.Tensor: The structural similarity index.
    """
    ssim = StructuralSimilarityIndexMeasure(return_full_image=True)
    _, ssim_idx_full_image = ssim(target, prediction)
    if mask is None:
        mask = torch.ones_like(ssim_idx_full_image)
    try:
        ssim_idx = ssim_idx_full_image[mask]
    except Exception as e:
        print(f"Error: {e}")
        if len(ssim_idx_full_image.shape) == 0:
            ssim_idx = torch.ones_like(mask) * ssim_idx_full_image
    return ssim_idx.mean().item()
