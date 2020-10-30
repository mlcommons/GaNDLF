# -*- coding: utf-8 -*-
"""
All Models in GANDLF are to be derived from this base class code
"""

import torch.nn as nn
import torch.nn.functional as F
import sys

class ModelBase(nn.Module):
    '''
    This is the base model class that all other architectures will need to derive from
    '''
    def __init__(self, n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer):
        """
        This defines all defaults that the model base uses
        """
        super(ModelBase, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_filters = base_filters
        self.n_dimensions = n_dimensions

        # based on dimensionality, the following need to defined:
        # convolution, batch_norm, instancenorm, dropout

        if self.n_dimensions == 2:
            self.Convolution = nn.Conv2d
        elif self.n_dimensions == 3:
            self.Convolution = nn.Conv3d
        else:
            sys.exit('Currently, only 2D or 3D datasets are supported.')

        none_list = ['none', None, 'None', 'regression']

        if final_convolution_layer == 'sigmoid':
            self.final_convolution_layer = F.sigmoid()

        elif final_convolution_layer == 'softmax':
            self.final_convolution_layer = F.softmax()

        elif final_convolution_layer in none_list:
            self.final_convolution_layer = None
        