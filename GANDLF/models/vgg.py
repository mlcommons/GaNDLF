# -*- coding: utf-8 -*-
"""
Modified from https://github.com/pytorch/vision.git
"""

import sys, math
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
            configuration (dict): A dictionary of configuration parameters for the model.
            parameters (dict) - overall parameters dictionary; parameters specific for DenseNet:
        """
        super(VGG, self).__init__(parameters)

        self.features = self.make_layers(
            configuration,
            self.n_channels,
        )

        # amp is not supported for vgg
        parameters["model"]["amp"] = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            self.GlobalAvgPool(),
            nn.ReLU(True),
            nn.Dropout(),
            # number of input features should be changed later, but works for all vgg right now
            nn.Linear(512, self.n_classes),
        )
        # Initialize weights
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
        out = self.features(x)
        out = self.classifier(out)
        if not self.final_convolution_layer is None:
            if self.final_convolution_layer == F.softmax:
                out = self.final_convolution_layer(out, dim=1)
            else:
                out = self.final_convolution_layer(out)
        return out

    def make_layers(self, cfg, in_channels):
        layers = []
        for v in cfg:
            if v == "M":
                layers += [self.MaxPool(kernel_size=2, stride=2)]
            else:
                conv = self.Conv(in_channels, v, kernel_size=3, padding=1)
                if self.norm_type in ["batch", "instance"]:
                    layers += [conv, self.Norm(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


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
    VGG model
    """

    def __init__(
        self,
        parameters,
    ):
        super(vgg11, self).__init__(parameters=parameters, configuration=cfg["A"])


class vgg13(VGG):
    """
    VGG model
    """

    def __init__(
        self,
        parameters,
    ):
        super(vgg13, self).__init__(parameters=parameters, configuration=cfg["B"])


class vgg16(VGG):
    """
    VGG model
    """

    def __init__(
        self,
        parameters,
    ):
        super(vgg16, self).__init__(parameters=parameters, configuration=cfg["D"])


class vgg19(VGG):
    """
    VGG model
    """

    def __init__(
        self,
        parameters,
    ):
        super(vgg19, self).__init__(parameters=parameters, configuration=cfg["E"])
