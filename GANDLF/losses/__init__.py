"""
All the losses are to be called from here
"""
from tkinter import Label

from .hybrid import DCCE, DCCE_Logits
from .regression import (
    CE,
    CEL,
    L1_loss,
    LabelSmoothingCrossEntropy,
    MSE_loss,
    SoftTargetCrossEntropy,
)
from .segmentation import KullbackLeiblerDivergence, MCD_log_loss, MCD_loss, MCT_loss

# global defines for the losses
global_losses_dict = {
    "dc": MCD_loss,
    "dice": MCD_loss,
    "dc_log": MCD_log_loss,
    "dice_log": MCD_log_loss,
    "dcce": DCCE,
    "dcce_logits": DCCE_Logits,
    "ce": CE,
    "mse": MSE_loss,
    "cel": CEL,
    "tversky": MCT_loss,
    "kld": KullbackLeiblerDivergence,
    "l1": L1_loss,
    "lsce": LabelSmoothingCrossEntropy,
    "stce": SoftTargetCrossEntropy,
}

