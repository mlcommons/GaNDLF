# -*- coding: utf-8 -*-
"""
Modified from https://github.com/pytorch/vision.git
"""
import sys, math
import torch.nn as nn
from .modelBase import get_final_layer

__all__ = [
    "POOL_TEST",
    "pool_test11",
    "pool_test11_bn",
    "pool_test13",
    "pool_test13_bn",
    "pool_test16",
    "pool_test16_bn",
    "pool_test19_bn",
    "pool_test19",
]


class POOL_TEST(nn.Module):
    """
    POOL_TEST model
    """

    def __init__(
        self,
        n_dimensions,
        features,
        inputFeaturesForClassifier,
        n_outputClasses,
        final_convolution_layer: str = "softmax",
    ):
        super(POOL_TEST, self).__init__()
        self.features = features
        self.final_convolution_layer = get_final_layer(final_convolution_layer)
        if n_dimensions == 2:
            self.Conv = nn.Conv2d
        elif n_dimensions == 3:
            self.Conv = nn.Conv3d
        else:
            sys.exit("Only 2D or 3D convolutions are supported.")

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(inputFeaturesForClassifier, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
            nn.ReLU(True),
            nn.Linear(10, n_outputClasses),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, self.Conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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


def pool_test11(n_dimensions=3):
    """POOL_TEST 11-layer model (configuration "A")"""
    return POOL_TEST(make_layers(cfg["A"], n_dimensions))


def pool_test11_bn(n_dimensions=3):
    """POOL_TEST 11-layer model (configuration "A") with batch normalization"""
    return POOL_TEST(make_layers(cfg["A"], n_dimensions, batch_norm=True))


def pool_test13(n_dimensions=3):
    """POOL_TEST 13-layer model (configuration "B")"""
    return POOL_TEST(make_layers(cfg["B"], n_dimensions=3))


def pool_test13_bn(n_dimensions=3):
    """POOL_TEST 13-layer model (configuration "B") with batch normalization"""
    return POOL_TEST(make_layers(cfg["B"], n_dimensions, batch_norm=True))


def pool_test16(n_dimensions=3):
    """POOL_TEST 16-layer model (configuration "D")"""
    return POOL_TEST(make_layers(cfg["D"], n_dimensions))


def pool_test16_bn(n_dimensions=3):
    """POOL_TEST 16-layer model (configuration "D") with batch normalization"""
    return POOL_TEST(make_layers(cfg["D"], n_dimensions, batch_norm=True))


def pool_test19(n_dimensions=3):
    """POOL_TEST 19-layer model (configuration "E")"""
    return POOL_TEST(make_layers(cfg["E"], n_dimensions))


def pool_test19_bn(n_dimensions=3):
    """POOL_TEST 19-layer model (configuration 'E') with batch normalization"""
    return POOL_TEST(make_layers(cfg["E"], n_dimensions, batch_norm=True))
