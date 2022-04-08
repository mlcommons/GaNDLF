"""
All the losses are to be called from here
"""
from .segmentation import MCD_loss, MCD_log_loss, MCT_loss, KullbackLeiblerDivergence
from .regression import CE, BCE, CEL, MSE_loss, L1_loss
from .hybrid import DCCE, DCCE_Logits


# global defines for the losses
global_losses_dict = {
    "dc": MCD_loss,
    "dice": MCD_loss,
    "dc_log": MCD_log_loss,
    "dice_log": MCD_log_loss,
    "dcce": DCCE,
    "dcce_logits": DCCE_Logits,
    "ce": CE,
    "bce": BCE,
    "mse": MSE_loss,
    "cel": CEL,
    "tversky": MCT_loss,
    "kld": KullbackLeiblerDivergence,
    "l1": L1_loss,
}
