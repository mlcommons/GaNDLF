# adapted from https://github.com/kenshohara/3D-ResNets-PyTorch

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .modelBase import ModelBase


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, Norm, Conv):
        """
        Constructor for _DenseLayer class.

        Args:
            num_input_features (int): Number of input channels to the layer.
            growth_rate (int): Number of output channels of each convolution operation in the layer.
            bn_size (int): Factor to scale the number of intermediate channels between the 1x1 and 3x3 convolutions.
            drop_rate (float): Probability of an element to be zeroed in the Dropout layer.
            Norm (torch.nn.Module): A normalization module from torch.nn, such as BatchNorm2d or InstanceNorm2d.
            Conv (torch.nn.Module): A convolution module from torch.nn, such as Conv2d or ConvTranspose2d.
        """
        # Call the constructor of the parent class
        super().__init__()

        # Add a batch normalization layer followed by a ReLU activation function and a 1x1 convolution layer
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

        # Add another batch normalization layer followed by a ReLU activation function and a 3x3 convolution layer
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

        # Set the dropout rate
        self.drop_rate = drop_rate

    def forward(self, x):
        """
        Forward pass through the _DenseLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor obtained after concatenating the input tensor and the new features obtained after the convolution operations and dropout.
        """
        # Perform forward pass through the layers
        new_features = super().forward(x)

        # If dropout rate is greater than 0, apply dropout to the new features
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )

        # Concatenate the input tensor with the new features and return the result
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
        """
        Constructor for _DenseBlock class.

        Args:
            num_layers (int): Number of dense layers to be added to the block.
            num_input_features (int): Number of input channels to the block.
            bn_size (int): Factor to scale the number of intermediate channels between the 1x1 and 3x3 convolutions in each dense layer.
            growth_rate (int): Number of output channels of each convolution operation in each dense layer.
            drop_rate (float): Probability of an element to be zeroed in the Dropout layer of each dense layer.
            norm (torch.nn.Module): A normalization module from torch.nn, such as BatchNorm2d or InstanceNorm2d, to be used in each dense layer.
            conv (torch.nn.Module): A convolution module from torch.nn, such as Conv2d or ConvTranspose2d, to be used in each dense layer.
        """
        # Call the constructor of the parent class
        super().__init__()

        # Add num_layers _DenseLayer objects to the block
        for i in range(num_layers):
            # Calculate the number of input features for the i-th dense layer
            num_input_features_i = num_input_features + i * growth_rate

            # Create an instance of _DenseLayer with the calculated number of input features and other parameters
            layer = _DenseLayer(
                num_input_features_i, growth_rate, bn_size, drop_rate, norm, conv
            )

            # Add the _DenseLayer object to the block
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
    def __init__(self, parameters: dict, block_config=(6, 12, 24, 16)):
        """
        Densenet-BC model class

        Args:
            block_config (list of 4 ints): how many layers in each pooling block
            parameters (dict): overall parameters dictionary; parameters specific for DenseNet:
                - growth_rate (int): how many filters to add each layer (k in paper)
                - num_init_features (int): the number of filters to learn in the first convolution layer
                - bn_size (int): multiplicative factor for number of bottle neck layers
                (i.e. bn_size * k features in the bottleneck layer)
                - drop_rate (float): dropout rate after each dense layer
                - num_classes (int): number of classification classes
                - final_convolution_layer (str): the final convolutional layer to use
                - norm_type (str): the normalization type to use
        """

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
        """
        Forward pass of the network

        Args:
            x (torch.Tensor) - the input tensor

        Returns:
            out (torch.Tensor) - the output tensor

        """
        # Pass the input tensor through the convolutional layers of the model
        features = self.features(x)

        # Apply a ReLU activation function to the feature maps
        out = F.relu(features, inplace=True)

        # Apply adaptive average pooling to the feature maps and flatten the resulting tensor
        out = self.AdaptiveAvgPool(self.output_size)(out).view(features.size(0), -1)

        # Pass the flattened tensor through the fully connected layers of the model
        out = self.classifier(out)

        # Apply the final convolutional operation, if specified
        if not self.final_convolution_layer is None:
            if self.final_convolution_layer == F.softmax:
                out = self.final_convolution_layer(out, dim=1)
            else:
                out = self.final_convolution_layer(out)

        # Return the output tensor
        return out


def densenet121(parameters):
    return DenseNet(parameters, block_config=(6, 12, 24, 16))


def densenet169(parameters):
    return DenseNet(parameters, block_config=(6, 12, 32, 32))


def densenet201(parameters):
    return DenseNet(parameters, block_config=(6, 12, 48, 32))


def densenet264(parameters):
    return DenseNet(parameters, block_config=(6, 12, 64, 48))
