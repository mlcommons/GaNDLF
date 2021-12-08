import torch

def GAN_loss(out, target):
    loss_fn = torch.nn.BCELoss()
    loss = loss_fn(out, target)
    return loss

def LSGAN_loss(out, target):
    loss = 0.5 * torch.mean((out-target)**2)
    return loss


