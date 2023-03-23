from .segmentation import MCD_loss
from .regression import CCE_Generic, CE, CE_Logits


def DCCE(predicted_mask, ground_truth, params):
    """
    Calculates the Dice-Cross-Entropy loss.

    Parameters
    ----------
    predicted_mask : torch.Tensor
        Predicted mask
    ground_truth : torch.Tensor
        Ground truth mask
    params : dict
        Dictionary of parameters

    Returns
    -------
    torch.Tensor
        Calculated loss
    """
    dcce_loss = MCD_loss(predicted_mask, ground_truth, params) + CCE_Generic(
        predicted_mask, ground_truth, params, CE
    )
    return dcce_loss


def DCCE_Logits(predicted_mask, ground_truth, params):
    """
    Calculates the Dice-Cross-Entropy loss using logits.

    Parameters
    ----------
    predicted_mask : torch.Tensor
        Predicted mask logits
    ground_truth : torch.Tensor
        Ground truth mask
    params : dict
        Dictionary of parameters

    Returns
    -------
    torch.Tensor
        Calculated loss
    """
    dcce_loss = MCD_loss(predicted_mask, ground_truth, params) + CCE_Generic(
        predicted_mask, ground_truth, params, CE_Logits
    )
    return dcce_loss
