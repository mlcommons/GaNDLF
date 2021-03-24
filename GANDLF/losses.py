import torch
import sys
from torch.nn import MSELoss


# Dice scores and dice losses   
def dice(inp, target):
    smooth = 1e-7
    iflat = inp.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth)) # 2 * intersection / union

def MCD(pm, gt, num_class, weights = None, ignore_class = None): 
    '''
    These weights should be the dice weights, not dice weights
    '''
    acc_dice = 0
    for i in range(0, num_class): # 0 is background
        calculate_dice_for_label = True
        if ignore_class is not None:
            if i == ignore_class:
                calculate_dice_for_label = False
        
        if calculate_dice_for_label:
            currentDice = dice(gt[:, i, ...], pm[:, i, ...])
            # currentDiceLoss = 1 - currentDice # subtract from 1 because this is a loss
            if weights is not None:
                currentDice = currentDice * weights[i]
            acc_dice += currentDice
    if weights is None:
        acc_dice /= num_class # we should not be considering 0
    return acc_dice

def MCD_loss(pm, gt, params): 
    '''
    These weights should be the penalty weights, not dice weights
    '''
    acc_dice_loss = 0
    num_class = params["model"]["num_classes"]

    if params['weights'] is not None:
        weights = params['weights']
    else:
        weights = None

    for i in range(0, params["model"]["num_classes"]): # 0 is background
        currentDice = dice(gt[:,i, ...], pm[:,i, ...])
        currentDiceLoss = 1 - currentDice # subtract from 1 because this is a loss
        if weights is not None:
            currentDiceLoss = currentDiceLoss * weights[i]
        acc_dice_loss += currentDiceLoss

    if weights is None:
        acc_dice_loss /= num_class # we should not be considering 0

    return acc_dice_loss

def MCD_loss_new(pm, gt, num_class, weights = None):    # compute the actual dice score
    dims = (1, 2, 3)
    eps = torch.finfo(torch.float32).eps
    intersection = torch.sum(pm * gt, dims)
    cardinality = torch.sum(pm + gt, dims)

    dice_score = (2. * intersection + eps) / (cardinality + eps)

    return torch.mean(-dice_score + 1.)

def MCD_log_loss(pm, gt, num_class, weights = None): 
    '''
    These weights should be the penalty weights, not dice weights
    '''
    acc_dice_loss = 0
    for i in range(0, num_class): # 0 is background
        currentDice = dice(gt[:, i, ...], pm[:, i, ...])
        currentDiceLoss = -torch.log(currentDice + torch.finfo(torch.float32).eps) # negative because we want positive losses
        if weights is not None:
            currentDiceLoss = currentDiceLoss * weights[i]
        acc_dice_loss += currentDiceLoss
    if weights is None:
        acc_dice_loss /= num_class # we should not be considering 0
    return acc_dice_loss

def CE(out,target):
    if bool(torch.sum(target) == 0): # contingency for empty mask
        return 0
    oflat = out.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    loss = torch.dot(-torch.log(oflat), tflat)/tflat.sum()
    return loss

def CCE(out, target, num_class, weights):
    acc_ce_loss = 0
    for i in range(0, num_class):
        curr_ce_loss = CE(out[:,i , ...], target[:, i, ...])
        if weights is not None:
            curr_ce_loss = curr_ce_loss * weights[i]
        acc_ce_loss += curr_ce_loss
    if weights is None:
        acc_ce_loss /= num_class 
    return acc_ce_loss
        
def DCCE(out,target, n_classes, weights):
    dcce_loss = MCD_loss(out,target, n_classes, weights) + CCE(out, target,n_classes, weights)
    return dcce_loss

def tversky(inp, target, alpha):
    smooth = 1e-7
    iflat = inp.view(-1)
    tflat = target.view(-1)
    intersection = (iflat*tflat).sum()
    fps = (iflat * (1-tflat)).sum()
    fns = ((1-iflat) * tflat).sum()
    denominator = intersection + (alpha*fps) + ((1-alpha)*fns) + smooth
    return (intersection+smooth)/denominator

def tversky_loss(inp, target, alpha):
    smooth = 1e-7
    iflat = inp.view(-1)
    tflat = inp.view(-1)
    intersection = (iflat*tflat).sum()
    fps = (inp * (1-target)).sum()
    fns = (inp * (1-target)).sum()
    denominator = intersection + (alpha*fps) + ((1-alpha)*fns) + smooth
    return 1 - ((intersection+smooth)/denominator)


def MCT_loss(inp, target, num_class, weights):
    acc_tv_loss = 0
    for i in range(0, num_class):
        acc_tv_loss += tversky_loss(inp[:, i, ...], target[:, i, ...]) * weights[i]
    acc_tv_loss /= (num_class-1)
    return acc_tv_loss

def MSE(output, label, reduction='mean', scaling_factor=1):
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
    label = label*scaling_factor
    loss = MSELoss(output, label, reduction=reduction)
    return loss

def MSE_loss(inp, target, num_classes, reduction = 'mean'):
    acc_mse_loss = 0
    for i in range(0, num_classes):
        acc_mse_loss += MSE(inp[:, i, ...], target[:, i, ...], reduction=reduction)
    acc_mse_loss/=num_classes
    return acc_mse_loss

def fetch_loss_function(loss_name, params):
    if loss_name == 'dc':
        loss_function = MCD_loss
    elif loss_name == 'dc_log':
        loss_function = MCD_log_loss
    elif loss_name == 'dcce':
        loss_function = DCCE
    elif loss_name == 'ce':
        loss_function = CE
    elif loss_name == 'mse':
        loss_function = MSE_loss
    else:
        print('WARNING: Could not find the requested loss function \'' + loss_name + '\' in the implementation, using dc, instead', file=sys.stderr)
        loss_name = 'dc'
        loss_function = MCD_loss
    return loss_function
