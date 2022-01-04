import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from . import networks
import sys





class ModelBase_Blocks(nn.Module):
    
    def __init__(self, ndim, norm_layer):
        super(ModelBase_Blocks, self).__init__()
        self.n_dimensions=ndim
        
            # based on dimensionality, the following need to defined:
            # convolution, batch_norm, instancenorm, dropout
        if ndim == 2:
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
            self.InstanceNorm = nn.InstanceNorm2d
            self.Dropout = nn.Dropout2d
            self.BatchNorm = nn.BatchNorm2d
            self.MaxPool = nn.MaxPool2d
            self.AvgPool = nn.AvgPool2d
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d
            self.AdaptiveMaxPool = nn.AdaptiveMaxPool2d
            self.Norm = norm_layer
            self.ReflectionPad = nn.ReflectionPad2d

        elif ndim == 3:
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
            self.InstanceNorm = nn.InstanceNorm3d
            self.Dropout = nn.Dropout3d
            self.BatchNorm = nn.BatchNorm3d
            self.MaxPool = nn.MaxPool3d
            self.AvgPool = nn.AvgPool3d
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool3d
            self.AdaptiveMaxPool = nn.AdaptiveMaxPool3d
            self.Norm = norm_layer
            self.ReflectionPad = nn.ReflectionPad2d

            
