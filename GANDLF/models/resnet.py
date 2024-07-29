import sys
import logging
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

from .modelBase import ModelBase
from GANDLF.utils import getBase2


class ResNet(ModelBase):
    """
    A class to define the Resnet model architecture.

    Args:
        parameters (dict): A dictionary of configuration parameters for the model.
        blockType (object): The block type to be used in the model.
        block_config (list): A list of integers, where each integer denotes the number of layers
        in each block of the model.
    """

    def __init__(self, parameters: dict, blockType, block_config):
        super(ResNet, self).__init__(parameters)

        # Check the patch size and get the number of allowed layers
        allowedLay = checkPatchDimensions(parameters["patch_size"], len(block_config))

        # Display warning message if patch size is not large enough for desired number of layers
        assert not (
            allowedLay != len(block_config) and allowedLay <= 0
        ), "The patch size is not large enough for the desired number of layers. It is expected that each dimension of the patch size is 2^(layers + 1)*i, where i is an integer greater than 2."
        if allowedLay != len(block_config) and allowedLay >= 1:
            logging.info(
                "The patch size is not large enough for the desired number of layers. It is expected that each dimension of the patch size is 2^(layers + 1)*i, where i is an integer greater than 2. Only the first %d layers will run."
                % allowedLay
            )

        block_config = block_config[:allowedLay]

        # Define defaults if not already defined
        if not ("num_init_features" in parameters):
            parameters["num_init_features"] = 64

        # Set output size based on number of dimensions
        if self.n_dimensions == 2:
            self.output_size = (1, 1)
        elif self.n_dimensions == 3:
            self.output_size = (1, 1, 1)

        # If normalization layer is not defined, use Batch Normalization
        if self.Norm is None:
            sys.stderr.write(
                "Warning: resnet is not defined without a normalization layer"
            )
            self.Norm = self.BatchNorm

        # Define first convolution layer with 7x7 conv stride 2, 2x2 pool stride 2
        self.features = [
            (
                "conv1",
                self.Conv(
                    in_channels=self.n_channels,
                    out_channels=parameters["num_init_features"],
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
            ),
            ("norm1", self.Norm(parameters["num_init_features"])),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", self.MaxPool(kernel_size=3, stride=2, padding=1)),
        ]

        # Add the first convolution layer to the sequential model
        self.features = nn.Sequential(OrderedDict(self.features))

        # make conv blocks
        num_features = parameters["num_init_features"]
        offset = num_features - num_features * 2**-1
        for i, num_lay in enumerate(block_config):
            block = blockType(
                num_in_feats=int(num_features * 2 ** (i - 1) + offset),
                num_out_feats=int(num_features * 2**i),
                num_layers=num_lay,
                Norm=self.Norm,
                Conv=self.Conv,
                num_block=i,
            )
            self.features.add_module("block{}".format(i + 1), block)
            offset = 0

        # final layer, fully connected -> number classes
        if blockType == _BottleNeckBlock:
            self.classifier = nn.Linear(4 * num_features * 2**i, self.n_classes)
        else:
            self.classifier = nn.Linear(num_features * 2**i, self.n_classes)

        # define starting weights
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
        Defines the computation performed at every forward pass.

        Args:
            x (tensor): input data

        Returns:
            tensor: output data
        """
        # Pass the input through the features layers
        features = self.features(x)

        # Perform adaptive average pooling
        out = self.AdaptiveAvgPool(self.output_size)(features).view(
            features.size(0), -1
        )

        # Pass the output through the classifier layer
        out = self.classifier(out)

        # Apply the final convolutional layer if it is defined
        if not self.final_convolution_layer is None:
            if self.final_convolution_layer == F.softmax:
                out = self.final_convolution_layer(out, dim=1)
            else:
                out = self.final_convolution_layer(out)

        # Return the output
        return out


class _BasicBlock(nn.Sequential):
    def __init__(self, num_in_feats, num_out_feats, num_layers, Norm, Conv, num_block):
        """
        Defines a basic block of layers with skip connection for ResNet.

        Args:
        num_in_feats (int): number of input features
        num_out_feats (int): number of output features
        num_layers (int): number of layers in the block
        Norm (nn.Module): normalization module (e.g., BatchNorm)
        Conv (nn.Module): convolution module (e.g., Conv2d, Conv3d)
        num_block (int): the index of the block within the network
        """
        super().__init__()

        # iterate through size of block
        num_feats = num_in_feats
        for i_lay in range(0, num_layers):
            # add basic layer
            layer = _BasicLayer(
                num_in_feats=num_feats,
                num_out_feats=num_out_feats,
                Norm=Norm,
                Conv=Conv,
                downsample=(num_block != 0) and (i_lay == 0),
            )
            self.add_module("layer{}".format(i_lay + 1), layer)
            num_feats = num_out_feats


class _BasicLayer(nn.Sequential):
    """
    A basic building block for a ResNet, consisting of two convolutional layers, and optionally a projection layer to match
    dimensions.

    Args:
        num_in_feats (int): number of input features
        num_out_feats (int): number of output features
        Norm (nn.Module): normalization layer
        Conv (nn.Module): convolutional layer
        downsample (bool): whether to downsample input
    """

    def __init__(self, num_in_feats, num_out_feats, Norm, Conv, downsample):
        super().__init__()

        # check if size needs to be changed
        self.sizing = num_out_feats != num_in_feats

        if self.sizing:
            # add module to project input to correct size
            self.add_module(
                "project",
                Conv(
                    num_in_feats,
                    num_out_feats,
                    kernel_size=1,
                    stride=1 + downsample,
                    padding=0,
                    bias=False,
                ),
            )
            self.add_module("projectNorm", Norm(num_out_feats))

        # first convolution layer
        self.add_module(
            "conv1",
            Conv(
                num_in_feats,
                num_out_feats,
                kernel_size=3,
                stride=1 + downsample,
                padding=1,
                bias=False,
            ),
        )
        self.add_module("norm1", Norm(num_out_feats))
        self.add_module("relu", nn.ReLU(inplace=True))

        # second convolution layer
        self.add_module(
            "conv2",
            Conv(
                num_out_feats,
                num_out_feats,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        self.add_module("norm2", Norm(num_out_feats))

    def forward(self, x):
        """
        Forward pass of a basic layer.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        identity = x

        # Apply CONV -> NORM -> RELU
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        # Apply CONV -> NORM
        out = self.conv2(out)
        out = self.norm2(out)

        if self.sizing:
            # project input to correct output size if needed
            identity = self.project(identity)
            identity = self.projectNorm(identity)

        out += identity
        out = self.relu(out)

        return out


class _BottleNeckBlock(nn.Sequential):
    def __init__(self, num_in_feats, num_out_feats, num_layers, Norm, Conv, num_block):
        """
        Initialize a `_BottleNeckBlock` object.

        Args:
            num_in_feats (int): The number of input features.
            num_out_feats (int): The number of output features.
            num_layers (int): The number of layers in the block.
            Norm (nn.Module): The normalization layer to be used.
            Conv (nn.Module): The convolutional layer to be used.
            num_block (int): The index of the current block in the entire network.

        """
        super().__init__()

        # calculate number of features for the first layer
        if num_block != 0:
            num_feats = 4 * num_in_feats
        else:
            num_feats = num_in_feats

        # iterate through layers in block
        for i_lay in range(0, num_layers):
            # add bottle neck layer
            layer = _BottleNeckLayer(
                num_in_feats=num_feats,
                num_out_feats=num_out_feats,
                Norm=Norm,
                Conv=Conv,
                downsample=(num_block != 0)
                and (
                    i_lay == 0
                ),  # set downsample flag for the first layer of non-first block
            )
            self.add_module("layer{}".format(i_lay + 1), layer)
            num_feats = (
                4 * num_out_feats
            )  # calculate number of features for the next layer


class _BottleNeckLayer(nn.Sequential):
    def __init__(
        self,
        num_in_feats,  # true input number
        num_out_feats,  # 3rd conv increases features --> true output number - 4*num_out_feats
        Norm,
        Conv,
        downsample,
    ):
        """
        Initialize a _BottleNeckLayer.

        Args:
            num_in_feats (int): number of input features
            num_out_feats (int): number of output features
            Norm (nn.Module): normalization module
            Conv (nn.Module): convolution module
            downsample (bool): downsample the input
        """
        super().__init__()

        # Determine whether size of input/output is changing.
        self.sizing = 4 * num_out_feats != num_in_feats

        # Add a module to project the input to the correct size if needed.
        if self.sizing:
            self.add_module(
                "project",
                Conv(
                    num_in_feats,
                    4 * num_out_feats,
                    kernel_size=1,
                    stride=1 + downsample,
                    padding=0,
                    bias=False,
                ),
            )
            self.add_module("projectNorm", Norm(4 * num_out_feats))

        # Add a 1x1 conv, a 3x3 conv and a 1x1 conv in that order.
        self.add_module(
            "conv1",
            Conv(
                num_in_feats,
                num_out_feats,
                kernel_size=1,
                stride=1 + downsample,
                padding=0,
                bias=False,
            ),
        )
        self.add_module("norm1", Norm(num_out_feats))
        self.add_module("relu", nn.ReLU(inplace=True))

        self.add_module(
            "conv2",
            Conv(
                num_out_feats,
                num_out_feats,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        self.add_module("norm2", Norm(num_out_feats))

        self.add_module(
            "conv3",
            Conv(
                num_out_feats,
                4 * num_out_feats,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        self.add_module("norm3", Norm(4 * num_out_feats))

    def forward(self, x):
        """
        Forward pass of a _BottleNeckLayer.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        identity = x

        # Apply CONV -> NORM -> RELU
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        # Apply CONV -> NORM -> RELU
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        # Apply CONV -> NORM
        out = self.conv3(out)
        out = self.norm3(out)

        # project input to correct output size if needed
        if self.sizing:
            identity = self.project(identity)
            identity = self.projectNorm(identity)

        out += identity
        out = self.relu(out)

        return out


def checkPatchDimensions(patch_size, numlay):
    """
    Check that the given patch size is compatible with the number of layers
    specified.

    Args:
        patch_size (int or tuple of ints): the patch size
        numlay (int): the number of layers

    Returns:
        numlay (int): the number of layers minus one if the patch size is incompatible;
            otherwise, the number of layers
    """
    # Convert the patch size to an array if it's an integer
    if isinstance(patch_size, int):
        patch_size_to_check = np.array(patch_size)
    else:
        patch_size_to_check = patch_size

    # If the patch is 2D, don't check divisibility of last dimension
    if patch_size_to_check[-1] == 1:
        patch_size_to_check = patch_size_to_check[:-1]

    # Check that each dimension of the patch size is divisible by 2^(numlay+1) and
    # is greater than or equal to 2^(numlay+2)
    if all(
        [
            x >= 2 ** (numlay + 2) and x % 2 ** (numlay + 1) == 0
            for x in patch_size_to_check
        ]
    ):
        return numlay
    else:
        # If the patch size is incompatible, compute the base 2 logarithm of
        # each dimension of the patch size and return the minimum value minus 1
        base2 = np.array([getBase2(x) for x in patch_size_to_check])
        remain = patch_size_to_check / 2**base2

        layers = np.where(remain == 1, base2 - 1, base2)
        numlay = np.min(layers) - 1
        return numlay


def resnet18(parameters):
    """
    Create a ResNet-18 model with the given parameters.

    Args:
        parameters (Namespace): the parameters for the model

    Returns:
        ResNet: the ResNet-18 model
    """
    return ResNet(parameters, _BasicBlock, block_config=(2, 2, 2, 2))


def resnet34(parameters):
    """
    Create a ResNet-34 model with the given parameters.

    Args:
        parameters (Namespace): the parameters for the model

    Returns:
        ResNet: the ResNet-34 model
    """
    return ResNet(parameters, _BasicBlock, block_config=(3, 4, 6, 3))


def resnet50(parameters):
    """
    Create a ResNet-50 model with the given parameters.

    Args:
        parameters (Namespace): the parameters for the model

    Returns:
        ResNet: the ResNet-50 model
    """
    return ResNet(parameters, _BottleNeckBlock, block_config=(3, 4, 6, 3))


def resnet101(parameters):
    """
    Create a ResNet-101 model with the given parameters.

    Args:
        parameters (Namespace): the parameters for the model

    Returns:
        ResNet: the ResNet-101 model
    """
    return ResNet(parameters, _BottleNeckBlock, block_config=(3, 4, 23, 3))


def resnet152(parameters):
    """
    Create a ResNet-152 model with the given parameters.

    Args:
        parameters (Namespace): the parameters for the model

    Returns:
        ResNet: the ResNet-152 model
    """
    return ResNet(parameters, _BottleNeckBlock, block_config=(3, 8, 36, 3))


def resnet200(parameters):
    """
    Create a ResNet-200 model with the given parameters.

    Args:
        parameters (Namespace): the parameters for the model

    Returns:
        ResNet: the ResNet-200 model
    """
    return ResNet(parameters, _BottleNeckBlock, block_config=(3, 24, 36, 3))
