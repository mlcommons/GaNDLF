"""
All the losses are to be called from here
"""
from .segmentation import MCD_loss, MCD_log_loss, KullbackLeiblerDivergence
from .regression import CE, CEL, MSE_loss, L1_loss
from .hybrid import DCCE, DCCE_Logits


# global defines for the metrics
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
    "kld": KullbackLeiblerDivergence,
    "l1": L1_loss,
}


def fetch_loss_function(loss_name, params):

    if isinstance(loss_name, dict):  # this is currently only happening for mse_torch
        # check for mse_torch
        loss_function = MSE_loss
    elif loss_name == "dc":
        loss_function = MCD_loss
    elif loss_name == "dc_log":
        loss_function = MCD_log_loss
    elif loss_name == "dcce":
        loss_function = DCCE
    elif loss_name == "dcce_logits":
        loss_function = DCCE_Logits
    elif loss_name == "ce":
        loss_function = CE
    elif loss_name == "mse":
        loss_function = MSE_loss
    elif loss_name == "cel":
        loss_function = cel
    elif loss_name == "kld":
        loss_function = KullbackLeiblerDivergence
    elif loss_name == "l1":
        loss_function = L1_loss
    else:
        print(
            "WARNING: Could not find the requested loss function '"
            + loss_name
            + "' in the implementation, using dc, instead",
            file=sys.stderr,
        )
        loss_name = "dc"
        loss_function = MCD_loss
    return loss_function