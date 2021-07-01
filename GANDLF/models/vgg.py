# -*- coding: utf-8 -*-
"""
Modified from https://github.com/pytorch/vision.git
"""

import sys, math
import torch.nn as nn
from .modelBase import get_final_layer
from GANDLF.models.seg_modules.average_pool import (
    GlobalAveragePooling3D,
    GlobalAveragePooling2D,
)


class VGG(nn.Module):
    """
    VGG model
    """

    def __init__(
        self,
        n_dimensions,
        features,
        n_outputClasses,
        final_convolution_layer: str = "softmax",
    ):
        """
        Initializer function for the VGG model

        Args:
            n_dimensions (int): The number of dimensions in the input data (2 or 3).
            features (int): The number of features to extract from the input data.
            n_outputClasses (int): The number of output classes.
            final_convolution_layer (str, optional): The final layer of the model. Defaults to "softmax".
        """
        super(VGG, self).__init__()
        self.features = features
        self.final_convolution_layer = get_final_layer(final_convolution_layer)
        if n_dimensions == 2:
            self.Conv = nn.Conv2d
            self.avg_pool = GlobalAveragePooling2D()
        elif n_dimensions == 3:
            self.Conv = nn.Conv3d
            self.avg_pool = GlobalAveragePooling3D()
        else:
            sys.exit("Only 2D or 3D convolutions are supported.")

        self.classifier = nn.Sequential(
            nn.Dropout(),
            self.avg_pool,
            nn.ReLU(True),
            nn.Dropout(),
            # number of input features should be changed later, but works for all vgg right now
            nn.Linear(
                512, n_outputClasses
            ),
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
        x = self.features(x)
        return self.classifier(x)


def make_layers(cfg, n_dimensions, in_channels, batch_norm=False):
    layers = []
    if n_dimensions == 2:
        Conv = nn.Conv2d
        MaxPool = nn.MaxPool2d
        BatchNorm = nn.BatchNorm2d
    elif n_dimensions == 3:
        Conv = nn.Conv3d
        MaxPool = nn.MaxPool3d
        BatchNorm = nn.BatchNorm3d
    for v in cfg:
        if v == "M":
            layers += [MaxPool(kernel_size=2, stride=2)]
        else:
            conv = Conv(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv, BatchNorm(v), nn.ReLU(inplace=True)]
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
