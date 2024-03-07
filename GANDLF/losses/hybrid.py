import torch

from .segmentation import MCD_loss, FocalLoss
from .regression import CCE_Generic, CE, CE_Logits


def DCCE(prediction: torch.Tensor, target: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Calculates the Dice-Cross-Entropy loss.

    Args:
        prediction (torch.Tensor): The predicted mask.
        target (torch.Tensor): The ground truth mask.
        params (dict): The parameters.

    Returns:
        torch.Tensor: The calculated loss.
    """
    dcce_loss = MCD_loss(prediction, target, params) + CCE_Generic(
        prediction, target, params, CE
    )
    return dcce_loss


def DCCE_Logits(
    prediction: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    """
    Calculates the Dice-Cross-Entropy loss using logits.

    Args:
        prediction (torch.Tensor): The predicted mask.
        target (torch.Tensor): The ground truth mask.
        params (dict): The parameters.

    Returns:
        torch.Tensor: The calculated loss.
    """
    dcce_loss = MCD_loss(prediction, target, params) + CCE_Generic(
        prediction, target, params, CE_Logits
    )
    return dcce_loss


def DC_Focal(
    prediction: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    """
    Calculates the Dice-Focal loss.

    Args:
        prediction (torch.Tensor): The predicted mask.
        target (torch.Tensor): The ground truth mask.
        params (dict): The parameters.

    Returns:
        torch.Tensor: The calculated loss.
    """
    return MCD_loss(prediction, target, params) + FocalLoss(prediction, target, params)
