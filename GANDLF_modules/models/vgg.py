# -*- coding: utf-8 -*-
"""
Modified from https://github.com/pytorch/vision.git
"""

import math
import torch.nn as nn
import torch.nn.functional as F

from .modelBase import ModelBase


class VGG(ModelBase):
    """
    VGG model
    """

    def __init__(
        self,
        parameters: dict,
        configuration,
    ):
        """
        Initializer function for the VGG model

        Args:
            parameters (dict) - overall parameters dictionary; parameters specific for DenseNet:
            configuration (dict): A dictionary of configuration parameters for the model.

        """
        super(VGG, self).__init__(parameters)

        # amp is not supported for vgg
        parameters["model"]["amp"] = False

        # Setup the feature extractor
        self.features = self.make_layers(configuration, self.n_channels)

        # Setup the classifier
        # Dev Note: number of input features for linear layer should be changed later,
        # but works for all vgg right now
        self.classifier = nn.Sequential(
            nn.Dropout(),
            self.GlobalAvgPool(),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, self.n_classes),
        )

        # Initialize weights, if convolutional use He initialization, if linear use Xavier initialization
        for m in self.modules():
            if isinstance(m, self.Conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            else:
                pass

    def forward(self, x):
        """
        Forward pass function of the VGG model.

        Args:
            x (tensor): Input tensor of shape (batch_size, n_channels, height, width).

        Returns:
            tensor: Output tensor of shape (batch_size, n_classes).
        """
        out = self.features(x)
        out = self.classifier(out)
        if not self.final_convolution_layer is None:
            if self.final_convolution_layer == F.softmax:
                # Apply softmax activation to the output tensor if specified in the configuration
                out = self.final_convolution_layer(out, dim=1)
            else:
                # Apply whatever final layer specified in the configuration to the output tensor
                out = self.final_convolution_layer(out)
        return out

    def make_layers(self, layer_config, input_channels):
        """
        Function to create convolutional layers for the VGG model based on the given layer configuration
        for VGG based models including VGG11, VGG13, VGG16, VGG19.

        Args:
            layer_config (list): A list containing the configuration of convolutional layers and max pooling layers
            in_channels (int): The number of input channels

        Returns:
            nn.Sequential: A sequential module containing the convolutional layers
        """
        layers = []
        for layer in layer_config:
            if layer == "M":
                # If you found M, then add a maxpool layer
                layers += [self.MaxPool(kernel_size=2, stride=2)]
            else:
                # Otherwise, add a convolutional layer with the number of channels
                conv = self.Conv(input_channels, layer, kernel_size=3, padding=1)
                if self.norm_type in ["batch", "instance"]:
                    layers += [conv, self.Norm(layer), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                input_channels = layer
        return nn.Sequential(*layers)


# Layer configuration for VGG models, as per the paper and M represents maxpool
# and the integers represent the number of channels in the convolutional layers
cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class vgg11(VGG):
    """
    A class representing the VGG11 model, which is a variant of the VGG model architecture.

    Inherits from the VGG class and specifies the configuration for the VGG11 architecture.
    """

    def __init__(
        self,
        parameters,
    ):
        """
        Initializes the VGG11 model with the given parameters.

        Args:
            parameters (dict): A dictionary containing the parameters for the VGG11 model.
        """
        super(vgg11, self).__init__(parameters=parameters, configuration=cfg["A"])


class vgg13(VGG):
    """
    A class representing the VGG13 model, which is a variant of the VGG model architecture.

    Inherits from the VGG class and specifies the configuration for the VGG13 architecture.
    """

    def __init__(
        self,
        parameters,
    ):
        """
        Initializes the VGG13 model with the given parameters.

        Args:
            parameters (dict): A dictionary containing the parameters for the VGG13 model.
        """
        super(vgg13, self).__init__(parameters=parameters, configuration=cfg["B"])


class vgg16(VGG):
    """
    A class representing the VGG16 model, which is a variant of the VGG model architecture.

    Inherits from the VGG class and specifies the configuration for the VGG16 architecture.
    """

    def __init__(
        self,
        parameters,
    ):
        """
        Initializes the VGG16 model with the given parameters.

        Args:
            parameters (dict): A dictionary containing the parameters for the VGG16 model.
        """
        super(vgg16, self).__init__(parameters=parameters, configuration=cfg["D"])


class vgg19(VGG):
    """
    A class representing the VGG19 model, which is a variant of the VGG model architecture.

    Inherits from the VGG class and specifies the configuration for the VGG19 architecture.
    """

    def __init__(
        self,
        parameters,
    ):
        """
        Initializes the VGG19 model with the given parameters.

        Args:
            parameters (dict): A dictionary containing the parameters for the VGG19 model.
        """
        super(vgg19, self).__init__(parameters=parameters, configuration=cfg["E"])
