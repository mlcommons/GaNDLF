# -*- coding: utf-8 -*-
"""All Models in GANDLF are to be derived from this base class code."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from GANDLF.utils import get_linear_interpolation_mode
from GANDLF.models.seg_modules.average_pool import (
    GlobalAveragePooling3D,
    GlobalAveragePooling2D,
)


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
        self.final_convolution_layer = self.get_final_layer(
            parameters["model"]["final_layer"]
        )

        self.linear_interpolation_mode = get_linear_interpolation_mode(
            self.n_dimensions
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
            self.GlobalAvgPool = GlobalAveragePooling2D
            self.Norm = self.get_norm_type(self.norm_type.lower(), self.n_dimensions)

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
            self.GlobalAvgPool = GlobalAveragePooling3D
            self.Norm = self.get_norm_type(self.norm_type.lower(), self.n_dimensions)

    def get_final_layer(self, final_convolution_layer):
        """
        This function gets the final layer of the model.

        Args:
            final_convolution_layer (str): The final layer of the model as a string.

        Returns:
            Functional: sigmoid, softmax, or None
        """
        none_list = [
            "none",
            None,
            "None",
            "regression",
            "classification_but_not_softmax",
            "logits",
            "classification_without_softmax",
        ]

        if final_convolution_layer in ["sigmoid", "sig"]:
            final_convolution_layer = torch.sigmoid

        elif final_convolution_layer in ["softmax", "soft"]:
            final_convolution_layer = F.softmax

        elif final_convolution_layer in none_list:
            final_convolution_layer = None

        return final_convolution_layer


    def get_norm_type(self, norm_type, dimensions):
        """
        This function gets the normalization type for the model.

        Args:
            norm_type (str): Normalization type as a string.
            dimensions (str): The dimensionality of the model.

        Returns:
            _InstanceNorm or _BatchNorm: The normalization type for the model.
        """
        if dimensions == 3:
            if norm_type == "batch":
                norm_type = nn.BatchNorm3d
            elif norm_type == "instance":
                norm_type = nn.InstanceNorm3d
            else:
                norm_type = None
        elif dimensions == 2:
            if norm_type == "batch":
                norm_type = nn.BatchNorm2d
            elif norm_type == "instance":
                norm_type = nn.InstanceNorm2d
            else:
                norm_type = None

        return norm_type
