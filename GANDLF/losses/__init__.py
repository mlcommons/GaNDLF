"""
All the losses are to be called from here
"""
from .segmentation import (
    MCD_loss,
    MCD_log_loss,
    MCT_loss,
    KullbackLeiblerDivergence,
    FocalLoss,
    MCC_loss,
    MCC_log_loss,
)
from .regression import CE, CEL, MSE_loss, L1_loss
from .hybrid import DCCE, DCCE_Logits, DC_Focal


# global defines for the losses
global_losses_dict = {
    "dc": MCD_loss,
    "dice": MCD_loss,
    "dc_log": MCD_log_loss,
    "dclog": MCD_log_loss,
    "dice_log": MCD_log_loss,
    "dicelog": MCD_log_loss,
    "mcc": MCC_loss,
    "mcc_log": MCC_log_loss,
    "mcclog": MCC_log_loss,
    "mathews": MCC_loss,
    "mathews_log": MCC_log_loss,
    "dcce": DCCE,
    "dcce_logits": DCCE_Logits,
    "ce": CE,
    "mse": MSE_loss,
    "cel": CEL,
    "tversky": MCT_loss,
    "kld": KullbackLeiblerDivergence,
    "l1": L1_loss,
    "focal": FocalLoss,
    "dc_focal": DC_Focal,
}
