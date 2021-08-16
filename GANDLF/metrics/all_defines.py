"""
All the metrics are to be called from here
"""
import torch
from GANDLF.losses import MSE_loss, cel
from .segmentation import multi_class_dice, hd100, hd95
from .regression import accuracy, F1_score, classification_accuracy


def identity(output, label, params):
    """
    Always returns 0

    Parameters
    ----------
    output : Tensor
        Output predicted generally by the network
    label : Tensor
        Required target label to match the output with

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    _, _, _ = output, label, params
    return torch.Tensor(0)


global_metrics_dict = {
    "dice": multi_class_dice,
    "accuracy": accuracy,
    "mse": MSE_loss,
    "hd95": hd95,
    "hausdorff95": hd100,
    "hd100": hd100,
    "hausdorff": hd100,
    "hausdorff100": hd100,
    "cel": cel,
    "f1_score": F1_score,
    "f1": F1_score,
    "classification_accuracy": classification_accuracy,
}
