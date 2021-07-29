import torch
import sys
from torch.nn import MSELoss, CrossEntropyLoss
from .utils import one_hot


# Dice scores and dice losses
def dice(inp, target):
    smooth = 1e-7
    iflat = inp.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    # 2 * intersection / union
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def cel(out, target, params):
    if len(target.shape) > 1 and target.shape[-1] == 1:
        target = torch.squeeze(target, -1)

    class_weights = None
    if params["class_weights"]:
        class_weights = torch.FloatTensor(list(params["class_weights"].values()))

        # more examples you have in the training data, the smaller the weight you have in the loss
        class_weights = 1.0 / class_weights

        class_weights = class_weights.float().to(params["device"])
    
    cel = CrossEntropyLoss(weight=class_weights)
    return cel(out, target)

def MCD(pm, gt, num_class, weights=None, ignore_class=None, loss_type=0):
    """
    These weights should be the dice weights, not dice weights
    loss_type:
        0: no loss, normal dice calculation
        1: dice loss, (1-dice)
        2: log dice, -log(dice)
    """
    acc_dice = 0
    for i in range(0, num_class):  # 0 is background
        calculate_dice_for_label = True
        if ignore_class is not None:
            if i == ignore_class:
                calculate_dice_for_label = False

        if calculate_dice_for_label:
            currentDice = dice(gt[:, i, ...], pm[:, i, ...])
            if loss_type == 1:
                currentDice = 1 - currentDice  # subtract from 1 because this is a loss
            elif loss_type == 2:
                # negative because we want positive losses
                currentDice = -torch.log(currentDice + torch.finfo(torch.float32).eps)
            if weights is not None:
                currentDice = currentDice * weights[i]
            acc_dice += currentDice
    if weights is None:
        acc_dice /= num_class  # we should not be considering 0
    return acc_dice


def MCD_loss(pm, gt, params):
    """
    These weights should be the penalty weights, not dice weights
    """
    gt = one_hot(gt, params["model"]["class_list"])
    return MCD(pm, gt, len(params["model"]["class_list"]), params["weights"], None, 1)


def MCD_loss_new(pm, gt, num_class, weights=None):  # compute the actual dice score
    dims = (1, 2, 3)
    eps = torch.finfo(torch.float32).eps
    intersection = torch.sum(pm * gt, dims)
    cardinality = torch.sum(pm + gt, dims)

    dice_score = (2.0 * intersection + eps) / (cardinality + eps)

    return torch.mean(-dice_score + 1.0)


def MCD_log_loss(pm, gt, params):
    """
    These weights should be the penalty weights, not dice weights
    """
    gt = one_hot(gt, params["model"]["class_list"])
    return MCD(pm, gt, len(params["model"]["class_list"]), params["weights"], None, 2)


def CE_Logits(out, target):
    iflat = out.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    loss = torch.nn.BCEWithLogitsLoss()
    loss_val = loss(iflat, tflat)
    return loss_val


def CE(out, target):
    iflat = out.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    loss = torch.nn.BCELoss()
    loss_val = loss(iflat, tflat)
    return loss_val


def CCE_Generic(out, target, params, CCE_Type):
    """
    Generic function to calculate CCE loss

    Args:
        out (torch.tensor): The predicted output value for each pixel. dimension: [batch, class, x, y, z].
        target (torch.tensor): The ground truth label for each pixel. dimension: [batch, class, x, y, z] factorial_class_list.
        params (dict): The parameter dictionary.
        CCE_Type (torch.nn): The CE loss function type.

    Returns:
        torch.tensor: The final loss value after taking multiple classes into consideration
    """

    acc_ce_loss = 0
    target = one_hot(target, params["model"]["class_list"]).type(out.dtype)
    for i in range(0, len(params["model"]["class_list"])):
        curr_ce_loss = CCE_Type(out[:, i, ...], target[:, i, ...])
        if params["weights"] is not None:
            curr_ce_loss = curr_ce_loss * params["weights"][i]
        acc_ce_loss += curr_ce_loss
    if params["weights"] is None:
        acc_ce_loss /= len(params["model"]["class_list"])
    return acc_ce_loss


def DCCE(pm, gt, params):
    dcce_loss = MCD_loss(pm, gt, params) + CCE_Generic(pm, gt, params, CE)
    return dcce_loss


def DCCE_Logits(pm, gt, params):
    dcce_loss = MCD_loss(pm, gt, params) + CCE_Generic(pm, gt, params, CE_Logits)
    return dcce_loss


def tversky(inp, target, alpha):
    smooth = 1e-7
    iflat = inp.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    fps = (iflat * (1 - tflat)).sum()
    fns = ((1 - iflat) * tflat).sum()
    denominator = intersection + (alpha * fps) + ((1 - alpha) * fns) + smooth
    return (intersection + smooth) / denominator


def tversky_loss(inp, target, alpha=1):
    tversky_val = tversky(inp, target, alpha)
    return 1 - tversky_val


def MCT_loss(inp, target, num_class, weights):
    acc_tv_loss = 0
    for i in range(0, num_class):
        acc_tv_loss += tversky_loss(inp[:, i, ...], target[:, i, ...]) * weights[i]
    acc_tv_loss /= num_class - 1
    return acc_tv_loss


def MSE(output, label, reduction="mean", scaling_factor=1):
    """
    Calculate the mean square error between the output variable from the network and the target

    Parameters
    ----------
    output : torch.Tensor
        The output generated usually by the network
    target : torch.Tensor
        The label for the corresponding Tensor for which the output was generated
    reduction : string, optional
        DESCRIPTION. The default is 'mean'.
    scaling_factor : integer, optional
        The scaling factor to multiply the label with

    Returns
    -------
    loss : torch.Tensor
        Computed Mean Squared Error loss for the output and label

    """
    scaling_factor = torch.as_tensor(scaling_factor)
    label = label.float()
    label = label * scaling_factor
    loss_fn = MSELoss(reduction=reduction)
    iflat = output.contiguous().view(-1)
    tflat = label.contiguous().view(-1)
    loss = loss_fn(iflat, tflat)
    return loss


def MSE_loss(inp, target, params):
    acc_mse_loss = 0
    # if inp.shape != target.shape:
    #     sys.exit('Input and target shapes are inconsistent')

    if inp.shape[0] == 1:
        acc_mse_loss += MSE(
            inp,
            target,
            reduction=params["loss_function"]["mse"]["reduction"],
            scaling_factor=params["scaling_factor"],
        )
        # for i in range(0, params["model"]["num_classes"]):
        #    acc_mse_loss += MSE(inp[i], target[i], reduction=params["loss_function"]['mse']["reduction"])
    else:
        for i in range(0, params["model"]["num_classes"]):
            acc_mse_loss += MSE(
                inp[:, i, ...],
                target[:, i, ...],
                reduction=params["loss_function"]["mse"]["reduction"],
                scaling_factor=params["scaling_factor"],
            )
    acc_mse_loss /= params["model"]["num_classes"]

    return acc_mse_loss


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
