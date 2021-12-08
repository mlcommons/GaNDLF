"""
All the losses are to be called from here
"""
from .segmentation import MCD_loss, MCD_log_loss, MCT_loss, KullbackLeiblerDivergence
from .regression import CE, CEL, MSE_loss, L1_loss
from .hybrid import DCCE, DCCE_Logits
from .gan import GAN_loss, LSGAN_loss


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
    "gan": GAN_loss,
    "lsgan": LSGAN_loss
}
