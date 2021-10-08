# -*- coding: utf-8 -*-
"""All Models in GANDLF are to be derived from this base class code."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_final_layer(final_convolution_layer):
    none_list = [
        "none",
        None,
        "None",
        "regression",
        "classification_but_not_softmax",
        "logits",
        "classification_without_softmax",
    ]

    if final_convolution_layer == "sigmoid":
        final_convolution_layer = torch.sigmoid

    elif final_convolution_layer == "softmax":
        final_convolution_layer = F.softmax

    elif final_convolution_layer in none_list:
        final_convolution_layer = None

    return final_convolution_layer


def get_norm_type(norm_type, dimensions):
    if dimensions == 3:
        if norm_type == "batch":
            norm_type = nn.BatchNorm3d
        elif norm_type == "instance":
            norm_type = nn.InstanceNorm3d
    elif dimensions == 2:
        if norm_type == "batch":
            norm_type = nn.BatchNorm2d
        elif norm_type == "instance":
            norm_type = nn.InstanceNorm2d

    return norm_type


class ModelBase(nn.Module):
    """
    This is the base model class that all other architectures will need to derive from
    """

    def __init__(self, parameters):
        """
        This defines all defaults that the model base uses

        Args:
            parameters (dict): This is a dictionary of all parameters that are needed for the model.
        """
        super(ModelBase, self).__init__()
        self.model_name = parameters["model"]["architecture"]
        self.n_dimensions = parameters["model"]["dimension"]
        self.n_channels = parameters["model"]["num_channels"]
        self.n_classes = parameters["model"]["num_classes"]
        self.base_filters = parameters["model"]["base_filters"]
        self.norm_type = parameters["model"]["norm_type"]
        self.patch_size = parameters["patch_size"]
        self.batch_size = parameters["batch_size"]
        self.amp = parameters["model"]["amp"]
        self.final_convolution_layer = get_final_layer(
            parameters["model"]["final_layer"]
        )

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
            self.Norm = get_norm_type(self.norm_type.lower(), self.n_dimensions)

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
            self.Norm = get_norm_type(self.norm_type.lower(), self.n_dimensions)
