# adapted from https://github.com/kenshohara/3D-ResNets-PyTorch

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .modelBase import ModelBase


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, Norm, Conv):
        super().__init__()
        self.add_module("norm1", Norm(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module(
            "conv1",
            Conv(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("norm2", Norm(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module(
            "conv2",
            Conv(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        num_layers,
        num_input_features,
        bn_size,
        growth_rate,
        drop_rate,
        norm,
        conv,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
                norm,
                conv,
            )
            self.add_module("denselayer{}".format(i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, Norm, Conv, AvgPool):
        super().__init__()
        self.add_module("norm", Norm(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            Conv(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("pool", AvgPool(kernel_size=2, stride=2))


class DenseNet(ModelBase):
    """Densenet-BC model class
    Args:
        block_config (list of 4 ints) - how many layers in each pooling block
        parameters (dict) - overall parameters dictionary; parameters specific for DenseNet:
            growth_rate (int) - how many filters to add each layer (k in paper)
            num_init_features (int) - the number of filters to learn in the first convolution layer
            bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
            drop_rate (float) - dropout rate after each dense layer
            num_classes (int) - number of classification classes
            final_convolution_layer (str) - the final convolutional layer to use
            norm_type (str) - the normalization type to use
    """

    def __init__(
        self,
        parameters: dict,
        block_config=(6, 12, 24, 16),
    ):

        super(DenseNet, self).__init__(parameters)

        # defining some defaults
        if not ("num_init_features" in parameters):
            parameters["num_init_features"] = 64
        if not ("growth_rate" in parameters):
            parameters["growth_rate"] = 32
        if not ("bn_size" in parameters):
            parameters["bn_size"] = 4
        if not ("drop_rate" in parameters):
            parameters["drop_rate"] = 0
        if not ("conv1_t_stride" in parameters):
            parameters["conv1_t_stride"] = 1
        if not ("conv1_t_size" in parameters):
            parameters["conv1_t_size"] = 7
        if not ("no_max_pool" in parameters):
            parameters["no_max_pool"] = False
        if self.Norm is None:
            sys.stderr.write(
                "Warning: densenet is not defined without a normalization layer"
            )
            self.Norm = self.BatchNorm

        if self.n_dimensions == 2:
            self.output_size = (1, 1)
            self.conv_stride = (parameters["conv1_t_stride"], 2)
        elif self.n_dimensions == 3:
            self.output_size = (1, 1, 1)
            self.conv_stride = (parameters["conv1_t_stride"], 2, 2)
        else:
            sys.exit("Only 2D or 3D convolutions are supported.")

        # First convolution
        self.features = [
            (
                "conv1",
                self.Conv(
                    self.n_channels,
                    parameters["num_init_features"],
                    kernel_size=parameters["conv1_t_size"],
                    stride=self.conv_stride,
                    padding=parameters["conv1_t_size"] // 2,
                    bias=False,
                ),
            ),
            ("norm1", self.Norm(parameters["num_init_features"])),
            ("relu1", nn.ReLU(inplace=True)),
        ]
        if not parameters["no_max_pool"]:
            self.features.append(
                ("pool1", self.MaxPool(kernel_size=3, stride=2, padding=1))
            )
        self.features = nn.Sequential(OrderedDict(self.features))

        # Each denseblock
        num_features = parameters["num_init_features"]
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=parameters["bn_size"],
                growth_rate=parameters["growth_rate"],
                drop_rate=parameters["drop_rate"],
                norm=self.Norm,
                conv=self.Conv,
            )
            self.features.add_module("denseblock{}".format(i + 1), block)
            num_features = num_features + num_layers * parameters["growth_rate"]
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    Norm=self.Norm,
                    Conv=self.Conv,
                    AvgPool=self.AvgPool,
                )
                self.features.add_module("transition{}".format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", self.Norm(num_features))

        for m in self.modules():
            if isinstance(m, self.Conv):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, self.Norm):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, self.n_classes)

        for m in self.modules():
            if isinstance(m, self.Conv):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, self.Norm):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.AdaptiveAvgPool(self.output_size)(out).view(features.size(0), -1)
        out = self.classifier(out)

        if not self.final_convolution_layer is None:
            if self.final_convolution_layer == F.softmax:
                out = self.final_convolution_layer(out, dim=1)
            else:
                out = self.final_convolution_layer(out)

        return out


def densenet121(parameters):
    return DenseNet(
        parameters,
        block_config=(6, 12, 24, 16),
    )


def densenet169(parameters):
    return DenseNet(parameters, block_config=(6, 12, 32, 32))


def densenet201(parameters):
    return DenseNet(parameters, block_config=(6, 12, 48, 32))


def densenet264(parameters):
    return DenseNet(parameters, block_config=(6, 12, 64, 48))
