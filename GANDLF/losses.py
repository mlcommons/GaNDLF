import torch 
from torch.nn import MSELoss
import math


# Dice scores and dice losses   
def dice(inp, target):
    smooth = 1e-7
    iflat = inp.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth)) # 2 * intersection / union

def MCD(pm, gt, num_class, weights = None): 
    acc_dice = 0
    for i in range(1, num_class): # 0 is background
        currentDiceLoss = dice(gt[:,i,:,:,:], pm[:,i,:,:,:]) # subtract from 1 because this is a loss
        if weights is not None:
            currentDiceLoss = currentDiceLoss * weights[i]
        acc_dice += currentDiceLoss
    acc_dice /= (num_class-1) # we should not be considering 0
    return acc_dice

def MCD_loss(pm, gt, num_class, weights = None): 
    return 1 - MCD(pm, gt, num_class, weights) 

def MCD_log_loss(pm, gt, num_class, weights = None): 
    return -torch.log(MCD(pm, gt, num_class, weights))

def CE(out,target):
    if bool(torch.sum(target) == 0): # contingency for empty mask
        return 0
    oflat = out.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    loss = torch.dot(-torch.log(oflat), tflat)/tflat.sum()
    return loss

def CCE(out, target, num_class, weights):
    acc_ce_loss = 0
    for i in range(1, num_class):
        acc_ce_loss += CE(out[:,i,:,:,:], target[:,i,:,:,:])
        if weights is not None:
            acc_ce_loss *= weights[i]
    acc_ce_loss /= (num_class-1)
    return acc_ce_loss
        
def DCCE(out,target, n_classes, weights):
    l = MCD_loss(out,target, n_classes, weights) + CCE(out, target,n_classes, weights)
    return l


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
    for i in range(1, num_class):
        acc_tv_loss += tversky_loss(inp[:,i,:,:,:], target[:,i,:,:,:]) * weights[i]
    acc_tv_loss /= (num_class-1)
    return acc_tv_loss

def MSE(inp, target, reduction = 'mean'):
    l = MSELoss(inp, target, reduction = reduction) # for reductions options, see https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
    return l

def MSE_loss(inp, target, num_classes, reduction = 'mean'):
    acc_mse_loss = 0
    for i in range(1, num_classes):
        acc_mse_loss += MSE(inp[:,i,:,:,:], target[:,i,:,:,:], reduction = reduction)
    acc_mse_loss/=num_classes
    return acc_mse_loss
