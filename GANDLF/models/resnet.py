import sys
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

from .modelBase import ModelBase

class ResNet(ModelBase):
    """
    Initializer function for the Resnet model

    Args:
        configuration (dict): A dictionary of configuration parameters for the model.
        parameters (dict) - overall parameters dictionary
    """

    def __init__(
        self,
        parameters: dict,
        blockType, # basic block or bottleneck
        block_config,
    ):
        super(ResNet, self).__init__(parameters)

        allowedLay = checkPatchDimensions(parameters["patch_size"], len(block_config))

        if allowedLay != len(block_config) and allowedLay >= 1:
            print(
                "The patch size is not large enough for desired number of layers.", 
                " It is expected that each dimension of the patch size is 2^(layers + 1)*i, where i is in a integer greater than 2.",
                "Only the first %d layers will run."%allowedLay
            )

        elif allowedLay != len(block_config) and allowedLay <= 0:
            sys.exit(
                "The patch size is not large enough for desired number of layers.", 
                " It is expected that each dimension of the patch size is 2^(layers + 1)*i, where i is in a integer greater than 2."
            )

        block_config = block_config[:allowedLay]

        # check/define defaults
        if not ("num_init_features" in parameters):
            parameters["num_init_features"] = 64
        if self.n_dimensions == 2:
            self.output_size = (1, 1)
        elif self.n_dimensions == 3:
            self.output_size = (1, 1, 1)
        else:
            sys.exit("Only 2D or 3D convolutions are supported.")
        if self.Norm is None:
            sys.stderr.write(
                "Warning: resnet is not defined without a normalization layer"
            )
            self.Norm = self.BatchNorm


        # first convolution: 7x7 conv stride 2, 2x2 pool stride 2
        self.features = [
            (
                "conv1",
                self.Conv(
                    in_channels = self.n_channels,
                    out_channels=parameters["num_init_features"],
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
            ),
            ("norm1", self.Norm(parameters["num_init_features"])),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", self.MaxPool(kernel_size=3, stride=2, padding=1))
        ]
        self.features = nn.Sequential(OrderedDict(self.features))

        
        # make conv blocks
        num_features = parameters["num_init_features"]
        offset = num_features - num_features*2**-1
        for i, num_lay in enumerate(block_config):
            block = blockType(
                num_in_feats = int(num_features*2**(i-1) + offset),
                num_out_feats = int(num_features*2**i),
                num_layers = num_lay,
                Norm = self.Norm,
                Conv = self.Conv,
                num_block = i,
            )
            self.features.add_module("block{}".format(i + 1), block)
            offset = 0

        # final layer, fully connected -> number classes
        if blockType == _BottleNeckBlock:
            self.classifier = nn.Linear(4*num_features*2**i, self.n_classes)
        else:
            self.classifier = nn.Linear(num_features*2**i, self.n_classes)

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
        features = self.features(x)
        out = self.AdaptiveAvgPool(self.output_size)(features).view(features.size(0), -1)
        out = self.classifier(out)

        if not self.final_convolution_layer is None:
            if self.final_convolution_layer == F.softmax:
                out = self.final_convolution_layer(out, dim=1)
            else:
                out = self.final_convolution_layer(out)

        return out



class _BasicBlock(nn.Sequential):
    def __init__(
        self,
        num_in_feats,
        num_out_feats,
        num_layers,
        Norm, Conv,
        num_block
    ):
        super().__init__()

        # iterate through size of block
        num_feats = num_in_feats
        for i_lay in range(0, num_layers):
            # add basic layer
            layer = _BasicLayer(
                    num_in_feats = num_feats,
                    num_out_feats = num_out_feats,
                    Norm = Norm,
                    Conv = Conv,
                    downsample = (num_block != 0) and (i_lay == 0)
            )
            self.add_module("layer{}".format(i_lay + 1), layer)
            num_feats = num_out_feats



class _BasicLayer(nn.Sequential):
    def __init__(
        self,
        num_in_feats,
        num_out_feats,
        Norm, Conv,
        downsample,
    ):
        super().__init__()

        self.sizing = (num_out_feats != num_in_feats) # true if size changes
        if self.sizing: # add module to project input to correct size
            self.add_module("project", 
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

        # 3x3 --> 3x3

        self.add_module(
            "conv1",
            Conv(
                num_in_feats,
                num_out_feats,
                kernel_size=3,
                stride=1+ downsample,
                padding=1,
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

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.sizing: # project input to correct output size if needed
            identity = self.project(identity)
            identity = self.projectNorm(identity)

        out += identity
        out = self.relu(out)
        return out


class _BottleNeckBlock(nn.Sequential):
    def __init__(
        self,
        num_in_feats,
        num_out_feats,
        num_layers,
        Norm, Conv,
        num_block
    ):
        super().__init__()

        # iterate through size of block
        if num_block != 0:
            num_feats = 4*num_in_feats
        else:
            num_feats = num_in_feats

        for i_lay in range(0, num_layers):
            # add basic layer
            layer = _BottleNeckLayer(
                    num_in_feats = num_feats,
                    num_out_feats = num_out_feats,
                    Norm = Norm,
                    Conv = Conv,
                    downsample = (num_block != 0) and (i_lay == 0)
            )
            self.add_module("layer{}".format(i_lay + 1), layer)
            num_feats = 4*num_out_feats


class _BottleNeckLayer(nn.Sequential):
    def __init__(
        self,
        num_in_feats, # true input number
        num_out_feats, # 3rd conv increases features --> true output number - 4*num_out_feats
        Norm, Conv,
        downsample,
    ):
        super().__init__()

        self.sizing = (4*num_out_feats != num_in_feats) # true if size changes

        if self.sizing: # add module to project input to correct size
            self.add_module("project", 
                Conv(
                    num_in_feats,
                    4*num_out_feats,
                    kernel_size=1,
                    stride=1+downsample,
                    padding=0,
                    bias=False,
                ),
            )
            self.add_module("projectNorm", Norm(4*num_out_feats))

        # 1x1 conv (k) -> 3x3 conv (k) -> 1x1 conv (4*k)

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
                4*num_out_feats,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        
        self.add_module("norm3", Norm(4*num_out_feats))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.sizing: # project input to correct output size if needed
            identity = self.project(identity)
            identity = self.projectNorm(identity)

        out += identity
        out = self.relu(out)
        return out

def checkPatchDimensions(patch_size, numlay):
    if isinstance(patch_size, int):
            patch_size_to_check = np.array(patch_size)
    else:
        patch_size_to_check = patch_size
    # for 2D, don't check divisibility of last dimension
    if patch_size_to_check[-1] == 1:
        patch_size_to_check = patch_size_to_check[:-1]

    if all([x >= 2**(numlay + 2) and x % 2**(numlay + 1) == 0 for x in patch_size_to_check]):
        return numlay
    else:
        # base2 = np.floor(np.log2(patch_size_to_check))
        base2 = np.array([getBase2(x) for x in patch_size_to_check])
        remain = patch_size_to_check / 2**base2 # check that at least 1

        layers = np.where(remain == 1, base2-1, base2)
        return int(np.min(layers) - 1)

def getBase2(num):
    base = 0
    while num%2 == 0:
        num = num /2
        base = base + 1
    return base

def resnet18(parameters):
    return ResNet(parameters, _BasicBlock, block_config=(2, 2, 2, 2))


def resnet34(parameters):
    return ResNet(parameters, _BasicBlock, block_config=(3, 4, 6, 3))


def resnet50(parameters):
    return ResNet(parameters, _BottleNeckBlock, block_config=(3, 4, 6, 3))


def resnet101(parameters):
    return ResNet(parameters, _BottleNeckBlock, block_config=(3, 4, 23, 3))


def resnet152(parameters):
    return ResNet(parameters, _BottleNeckBlock, block_config=(3, 8, 36, 3))

