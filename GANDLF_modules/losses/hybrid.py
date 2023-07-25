import torch

from .segmentation import MCD_loss, FocalLoss
from .regression import CCE_Generic, CE, CE_Logits


def DCCE(predicted_mask, ground_truth, params) -> torch.Tensor:
    """
    Calculates the Dice-Cross-Entropy loss.

    Args:
        predicted_mask (torch.Tensor): The predicted mask.
        ground_truth (torch.Tensor): The ground truth mask.
        params (dict): The parameters.

    Returns:
        torch.Tensor: The calculated loss.
    """
    dcce_loss = MCD_loss(predicted_mask, ground_truth, params) + CCE_Generic(
        predicted_mask, ground_truth, params, CE
    )
    return dcce_loss


def DCCE_Logits(predicted_mask, ground_truth, params):
    """
    Calculates the Dice-Cross-Entropy loss using logits.

    Args:
        predicted_mask (torch.Tensor): The predicted mask.
        ground_truth (torch.Tensor): The ground truth mask.
        params (dict): The parameters.

    Returns:
        torch.Tensor: The calculated loss.
    """
    dcce_loss = MCD_loss(predicted_mask, ground_truth, params) + CCE_Generic(
        predicted_mask, ground_truth, params, CE_Logits
    )
    return dcce_loss


def DC_Focal(predicted_mask, ground_truth, params):
    """
    Calculates the Dice-Focal loss.

    Args:
        predicted_mask (torch.Tensor): The predicted mask.
        ground_truth (torch.Tensor): The ground truth mask.
        params (dict): The parameters.

    Returns:
        torch.Tensor: The calculated loss.
    """
    return MCD_loss(predicted_mask, ground_truth, params) + FocalLoss(
        predicted_mask, ground_truth, params
    )
