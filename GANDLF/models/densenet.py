# adapted from https://github.com/kenshohara/3D-ResNets-PyTorch

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from GANDLF.models.modelBase import get_final_layer


class _DenseLayer(nn.Sequential):
    def __init__(
        self, num_input_features, growth_rate, bn_size, drop_rate, BatchNorm, Conv
    ):
        super().__init__()
        self.add_module("norm1", BatchNorm(num_input_features))
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
        self.add_module("norm2", BatchNorm(bn_size * growth_rate))
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
        batch_norm,
        conv,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
                batch_norm,
                conv,
            )
            self.add_module("denselayer{}".format(i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(
        self, num_input_features, num_output_features, BatchNorm, Conv, AvgPool
    ):
        super().__init__()
        self.add_module("norm", BatchNorm(num_input_features))
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


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(
        self,
        num_channels=3,
        num_dimensions=3,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        num_classes=1000,
        final_convolution_layer=None,
    ):

        super().__init__()

        if num_dimensions == 2:
            self.Conv = nn.Conv2d
            self.MaxPool = nn.MaxPool2d
            self.BatchNorm = nn.BatchNorm2d
            self.AvgPool = nn.AvgPool2d
            self.adaptive_avg_pool = F.adaptive_avg_pool2d
            self.output_size = (1, 1)
            self.conv_stride = (conv1_t_stride, 2)
        elif num_dimensions == 3:
            self.Conv = nn.Conv3d
            self.MaxPool = nn.MaxPool3d
            self.BatchNorm = nn.BatchNorm3d
            self.AvgPool = nn.AvgPool3d
            self.adaptive_avg_pool = F.adaptive_avg_pool3d
            self.output_size = (1, 1, 1)
            self.conv_stride = (conv1_t_stride, 2, 2)
        else:
            sys.exit("Only 2D or 3D convolutions are supported.")

        # First convolution
        self.features = [
            (
                "conv1",
                self.Conv(
                    num_channels,
                    num_init_features,
                    kernel_size=conv1_t_size,
                    stride=self.conv_stride,
                    padding=conv1_t_size // 2,
                    bias=False,
                ),
            ),
            ("norm1", self.BatchNorm(num_init_features)),
            ("relu1", nn.ReLU(inplace=True)),
        ]
        if not no_max_pool:
            self.features.append(
                ("pool1", self.MaxPool(kernel_size=3, stride=2, padding=1))
            )
        self.features = nn.Sequential(OrderedDict(self.features))

        self.final_convolution_layer = get_final_layer(final_convolution_layer)

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                batch_norm=self.BatchNorm,
                conv=self.Conv,
            )
            self.features.add_module("denseblock{}".format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    BatchNorm=self.BatchNorm,
                    Conv=self.Conv,
                    AvgPool=self.AvgPool,
                )
                self.features.add_module("transition{}".format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", self.BatchNorm(num_features))

        for m in self.modules():
            if isinstance(m, self.Conv):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, self.BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, self.Conv):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, self.BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.adaptive_avg_pool(out, output_size=self.output_size).view(
            features.size(0), -1
        )
        out = self.classifier(out)

        if not self.final_convolution_layer is None:
            if self.final_convolution_layer == F.softmax:
                out = self.final_convolution_layer(out, dim=1)
            else:
                out = self.final_convolution_layer(out)

        return out


def generate_model(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet(
            num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs
        )
    elif model_depth == 169:
        model = DenseNet(
            num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs
        )
    elif model_depth == 201:
        model = DenseNet(
            num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs
        )
    elif model_depth == 264:
        model = DenseNet(
            num_init_features=64, growth_rate=32, block_config=(6, 12, 64, 48), **kwargs
        )

    return model


def densenet121(parameters):
    return generate_model(
        121,
        num_classes=parameters["model"]["num_classes"],
        num_dimensions=parameters["model"]["dimension"],
        num_channels=parameters["model"]["num_channels"],
        final_convolution_layer=parameters["model"]["final_layer"],
    )


def densenet169(parameters):
    return generate_model(
        169,
        num_classes=parameters["model"]["num_classes"],
        num_dimensions=parameters["model"]["dimension"],
        num_channels=parameters["model"]["num_channels"],
        final_convolution_layer=parameters["model"]["final_layer"],
    )


def densenet201(parameters):
    return generate_model(
        201,
        num_classes=parameters["model"]["num_classes"],
        num_dimensions=parameters["model"]["dimension"],
        num_channels=parameters["model"]["num_channels"],
        final_convolution_layer=parameters["model"]["final_layer"],
    )


def densenet264(parameters):
    return generate_model(
        264,
        num_classes=parameters["model"]["num_classes"],
        num_dimensions=parameters["model"]["dimension"],
        num_channels=parameters["model"]["num_channels"],
        final_convolution_layer=parameters["model"]["final_layer"],
    )
