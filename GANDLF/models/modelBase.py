# -*- coding: utf-8 -*-
"""All Models in GANDLF are to be derived from this base class code."""

import torch.nn as nn

from acsconv.converters import ACSConverter, Conv3dConverter, SoftACSConverter

from GANDLF.utils import get_linear_interpolation_mode
from GANDLF.utils.generic import checkPatchDimensions
from GANDLF.utils.modelbase import get_modelbase_final_layer
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
        if "num_classes" in parameters["model"]:
            self.n_classes = parameters["model"]["num_classes"]
        else:
            self.n_classes = len(parameters["model"]["class_list"])
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

        self.sigmoid_input_multiplier = parameters["model"].get(
            "sigmoid_input_multiplier", 1.0
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
            self.converter = None

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

            # define 2d to 3d model converters
            converter_type = parameters["model"].get("converter_type", "soft").lower()
            self.converter = SoftACSConverter
            if converter_type == "acs":
                self.converter = ACSConverter
            elif converter_type == "conv3d":
                self.converter = Conv3dConverter

        else:
            raise ValueError(
                "GaNDLF only supports 2D and 3D computations. {}D computations are not currently supported".format(
                    self.n_dimensions
                )
            )

    def get_final_layer(self, final_convolution_layer):
        return get_modelbase_final_layer(final_convolution_layer)

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

    def model_depth_check(self, parameters):
        """
        This function checks if the patch size is large enough for the model.

        Args:
            parameters (dict): The entire set of parameters for the model.

        Returns:
            int: The model depth to use.

        Raises:
            AssertionError: If the patch size is not large enough for the model.
        """
        model_depth = checkPatchDimensions(
            parameters["patch_size"], numlay=parameters["model"]["depth"]
        )

        common_msg = "The patch size is not large enough for desired depth. It is expected that each dimension of the patch size is divisible by 2^i, where i is in a integer greater than or equal to 2."
        assert model_depth >= 2, common_msg

        if model_depth != parameters["model"]["depth"] and model_depth >= 2:
            print(common_msg + " Only the first %d layers will run." % model_depth)

        return model_depth
