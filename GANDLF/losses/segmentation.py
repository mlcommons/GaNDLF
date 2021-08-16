import torch
from GANDLF.utils import one_hot


# Dice scores and dice losses
def dice(output, label):
    """
    This function computes a dice score between two tensors

    Parameters
    ----------
    output : Tensor
        Output predicted generally by the network
    label : Tensor
        Required target label to match the output with

    Returns
    -------
    Tensor
        Computed Dice Score

    """
    smooth = 1e-7
    iflat = output.contiguous().view(-1)
    tflat = label.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


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


def KullbackLeiblerDivergence(mu, logvar, params=None):
    loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return loss.mean()
