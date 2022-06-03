from .regression import CE, CCE_Generic, CE_Logits
from .segmentation import MCD_loss


def DCCE(pm, gt, params):
    dcce_loss = MCD_loss(pm, gt, params) + CCE_Generic(pm, gt, params, CE)
    return dcce_loss


def DCCE_Logits(pm, gt, params):
    dcce_loss = MCD_loss(pm, gt, params) + CCE_Generic(pm, gt, params, CE_Logits)
    return dcce_loss
