# -*- coding: utf-8 -*-
"""All Models in GANDLF are to be derived from this base class code."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def get_final_layer(final_convolution_layer):
    none_list = ["none", None, "None", "regression", "classification_but_not_softmax"]

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
        self,
        n_dimensions,
        n_channels,
        n_classes,
        base_filters,
        norm_type,
        final_convolution_layer,
    ):
        """
        This defines all defaults that the model base uses

        Args:
            n_dimensions (int): The number of dimensions for the model to use - defines computational dimensions.
            n_channels (int): The number of channels for the model to use.
            n_classes (int): The number of output classes (used for segmentation).
            base_filters (int): The number of filters for the first convolutional layer.
            norm_type (str): The normalization type; can be 'instance' or 'batch'
            final_convolution_layer (str): The final layer of the model
        """
        super(ModelBase, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_filters = base_filters
        self.n_dimensions = n_dimensions
        self.norm_type = norm_type

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
            if self.norm_type.lower() == "batch":
                self.Norm = nn.BatchNorm2d
            elif self.norm_type.lower() == "instance":
                self.Norm = nn.InstanceNorm2d
            else:
                sys.exit("Currently, 'norm_type' supports only batch or instance.")

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
            if self.norm_type.lower() == "batch":
                self.Norm = nn.BatchNorm3d
            elif self.norm_type.lower() == "instance":
                self.Norm = nn.InstanceNorm3d
            else:
                sys.exit("Currently, 'norm_type' supports only batch or instance.")

        self.final_convolution_layer = get_final_layer(final_convolution_layer)
