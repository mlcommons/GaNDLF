# -*- coding: utf-8 -*-
"""
All Models in GANDLF are to be derived from this base class code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def get_final_layer(final_convolution_layer):
    none_list = ["none", None, "None", "regression"]

    if final_convolution_layer == "sigmoid":
        final_convolution_layer = torch.sigmoid

    elif final_convolution_layer == "softmax":
        final_convolution_layer = F.softmax

    elif final_convolution_layer in none_list:
        final_convolution_layer = None

    return final_convolution_layer


class ModelBase(nn.Module):
    """
    This is the base model class that all other architectures will need to derive from
    """

    def __init__(
        self, n_dimensions, n_channels, n_classes, base_filters, norm_type, final_convolution_layer
    ):
        """
        This defines all defaults that the model base uses
        """
        super(ModelBase, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_filters = base_filters
        self.n_dimensions = n_dimensions
        self.norm_type = norm_type
        if self.norm_type.lower() == "batch":
            if self.n_dimensions == 2:
                self.norm = nn.BatchNorm2d
            else:
                self.norm = nn.BatchNorm3d
        elif self.norm_type.lower() == "instance":
            if self.n_dimensions == 3:
                self.norm = nn.InstanceNorm2d
            else:
                self.norm = nn.InstanceNorm3d
        else:
            self.norm = None

        # based on dimensionality, the following need to defined:
        # convolution, batch_norm, instancenorm, dropout

        if self.n_dimensions == 2:
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
            self.InstanceNorm = nn.InstanceNorm2d
            self.Dropout = nn.Dropout2d
            self.BatchNorm = nn.BatchNorm2d
            self.MaxPool = nn.MaxPool2d
            self.AvgPool = nn.AvgPool2d
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d
            self.AdaptiveMaxPool = nn.AdaptiveMaxPool2d

        elif self.n_dimensions == 3:
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
            self.InstanceNorm = nn.InstanceNorm3d
            self.Dropout = nn.Dropout3d
            self.BatchNorm = nn.BatchNorm3d
            self.MaxPool = nn.MaxPool3d
            self.AvgPool = nn.AvgPool3d
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool3d
            self.AdaptiveMaxPool = nn.AdaptiveMaxPool3d
        else:
            sys.exit("Currently, only 2D or 3D datasets are supported.")

        self.final_convolution_layer = get_final_layer(final_convolution_layer)
