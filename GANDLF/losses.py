import numpy as np
import torch 


# Dice scores and dice losses   
def dice_loss(inp, target):
    smooth = 1e-7
    iflat = inp.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

def MCD_loss(pm, gt, num_class):
    acc_dice_loss = 0
    for i in range(1,num_class):
        acc_dice_loss += dice_loss(gt[:,i,:,:,:],pm[:,i,:,:,:]) 
    acc_dice_loss = acc_dice_loss/(num_class-1)
    return acc_dice_loss

# Setting up the Evaluation Metric
def dice(out, target):
    smooth = 1e-7
    oflat = out.view(-1)
    tflat = target.view(-1)
    intersection = (oflat * tflat).sum()
    return (2*intersection+smooth)/(oflat.sum()+tflat.sum()+smooth)


def CE(out,target):
    oflat = out.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    loss = torch.dot(-torch.log(oflat), tflat)/tflat.sum()
    return loss

def CCE(out, target, num_class):
    acc_ce_loss = 0
    for i in range(num_class):
        acc_ce_loss += CE(out[:,i,:,:,:],target[:,i,:,:,:])
    acc_ce_loss /= num_class
    return acc_ce_loss
        

def DCCE(out,target, n_classes):
    l = MCD_loss(out,target, n_classes) + CCE(out,target,n_classes)
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


def MCT_loss(inp, target, num_class):
    acc_tv_loss = 0
    for i in range(0,num_class):
        acc_tv_loss += tversky_loss(inp[:,i,:,:,:],target[:,i,:,:,:])
    acc_tv_loss/= num_class
    return acc_tv_loss

def MSE(inp,target):
    iflat = inp.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    num = len(iflat)
    loss = (iflat - tflat)*(iflat - tflat)
    loss = loss.sum()
    loss = loss/num
    return loss
    
def MSE_loss(inp,target,num_classes):
    acc_mse_loss = 0
    for i in range(0,num_classes):
        acc_mse_loss += MSE(inp[:,i,:,:,:], target[:,i,:,:,:])
    acc_mse_loss/=num_classes
    return acc_mse_loss
    
def MCD_MSE_loss(inp,target,num_classes):
    l = MCD_loss(inp,target,num_classes) + 0.1*MSE_loss(inp,target,num_classes)
    return l



