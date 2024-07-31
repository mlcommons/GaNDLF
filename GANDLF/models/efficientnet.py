import sys
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import numpy as np

from .modelBase import ModelBase


class _MBConv1(nn.Sequential):
    def __init__(
        self,
        num_in_feats,
        num_out_feats,
        kernel_size,
        stride,
        output_size,
        reduction,  # reduction factor for squeeze excitation
        Norm,
        Conv,
        Pool,
    ):
        """
        Defines a Single MBConv block in the EfficientNet model.

        Args:
            num_in_feats: Number of input features.
            num_out_feats: Number of output features.
            kernel_size: Size of the convolutional kernel.
            stride: Stride of the convolutional layer.
            output_size: Size of the output tensor.
            reduction: Reduction factor for squeeze excitation.
            Norm: Normalization layer for the MBConv block.
            Conv: Convolutional layer for the MBConv block.
            Pool: Pooling layer for the squeeze excitation block.

        Returns:
            Output tensor after passing through the MBConv block.

        """
        super().__init__()

        # depthwise conv -> batch norm -> swish
        # SE
        # conv -> batch norm

        self.add_module(
            "depthconv1",
            Conv(
                num_in_feats,
                num_in_feats,
                groups=num_in_feats,
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) / 2),
                bias=False,
            ),
        )
        self.add_module("norm1", Norm(num_in_feats))
        self.add_module("silu", nn.SiLU(inplace=True))

        # ADD SQUEEZE EXCITATION; no change in dimensions
        self.add_module(
            "squeeze", _SqueezeExcitation(num_in_feats, reduction, output_size, Pool)
        )

        self.add_module(
            "conv2",
            Conv(
                num_in_feats,
                num_out_feats,
                kernel_size=1,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                bias=False,
            ),
        )

        self.add_module("norm2", Norm(num_out_feats))

    def forward(self, x):
        """
        Sequentially passes the input tensor through depthwise convolution -> batch normalization -> swish activation ->
        squeeze excitation -> convolution -> batch normalization and returns the output tensor.

        Args:
            x (torch.Tensor): Input tensor for the MBConv block.

        Returns:
            out (torch.Tensor): Output tensor after passing through the MBConv block.
        """
        out = self.depthconv1(x)
        out = self.norm1(out)
        out = self.silu(out)

        out = self.squeeze(out)

        out = self.conv2(out)
        out = self.norm2(out)

        return out


class _MBConv6(nn.Sequential):
    def __init__(
        self,
        num_in_feats,
        num_out_feats,
        kernel_size,
        stride,
        output_size,
        Norm,
        Conv,
        Pool,
        reduction,  # reduction factor for squeeze excitation
    ):
        """
        A class representing a single MobileNetV3 block with expansion factor 6. Refer to https://arxiv.org/abs/1905.02244 for more.

        Args:
            num_in_feats (int): The number of input features/channels.
            num_out_feats (int): The number of output features/channels.
            kernel_size (int): The size of the kernel/convolutional filter.
            stride (int): The stride of the convolutional operation.
            output_size (int): The output size of the block.
            Norm (torch.nn.Module): A normalization layer to be used in the block.
            Conv (torch.nn.Module): A convolutional layer to be used in the block.
            Pool (torch.nn.Module): A pooling layer to be used in the squeeze-excitation block.
            reduction (float): The reduction factor for the squeeze-excitation block.
        """
        super().__init__()

        # conv -> batch norm --> swish
        # depthwise conv -> batch norm --> swish
        # SE
        # conv -> batch norm

        self.add_module(
            "conv1",
            Conv(
                num_in_feats,
                6 * num_in_feats,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        self.add_module("norm1", Norm(6 * num_in_feats))
        self.add_module("silu", nn.SiLU(inplace=True))

        self.add_module(
            "depthconv1",
            Conv(
                6 * num_in_feats,
                6 * num_in_feats,
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) / 2),
                bias=False,
            ),
        )
        self.add_module("norm2", Norm(6 * num_in_feats))

        # ADD SQUEEZE EXCITATION; no change in dimensions
        self.add_module(
            "squeeze",
            _SqueezeExcitation(6 * num_in_feats, reduction, output_size, Pool),
        )

        self.add_module(
            "conv2",
            Conv(
                6 * num_in_feats,
                num_out_feats,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        self.add_module("norm3", Norm(num_out_feats))

    def forward(self, x):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            torch.Tensor: The output tensor from the model.
        """
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.silu(out)

        out = self.depthconv1(out)
        out = self.norm2(out)
        out = self.silu(out)

        out = self.squeeze(out)

        out = self.conv2(out)
        out = self.norm3(out)

        return out


class _SqueezeExcitation(nn.Sequential):
    def __init__(
        self,
        num_in_feats,
        reduction,  # reduce dimension to int(num_in_feats / reduction)
        output_size,
        Pool,
    ):
        """
        Squeeze-and-Excitation block that recalibrates channel-wise feature responses

        Args:
            num_in_feats (int): Number of input features
            reduction (int): Reduction factor for number of hidden features
            output_size (tuple): Tuple containing the size of the output tensor
            Pool (nn.Module): Pooling operation
        """
        super().__init__()

        # global avg pool
        # FC --> ReLU
        # FC --> Sigmoid

        self.add_module("FC1", nn.Linear(num_in_feats, int(num_in_feats / reduction)))
        self.add_module("relu", nn.ReLU(inplace=True))

        self.add_module("FC2", nn.Linear(int(num_in_feats / reduction), num_in_feats))
        self.add_module("sigmoid", nn.Sigmoid())
        self.Pool = Pool
        self.output_size = output_size

    def forward(self, x):
        """
        Forward pass for the Squeeze-and-Excitation block

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, H, W]

        Returns:
            torch.Tensor: Output tensor of the same shape as the input tensor
        """
        out = self.Pool(self.output_size)(x).view(x.size(0), -1)
        out = self.FC1(out)
        out = self.relu(out)
        out = self.FC2(out)
        out = self.sigmoid(out)

        dims = list(x.size())
        dims[2:] = self.output_size
        return x * out.view(dims).expand_as(x)


DEFAULT_BLOCKS = [
    {
        "block_type": _MBConv1,
        "kernel": 3,
        "output_size": 16,
        "num_layers": 1,
        "stride": 1,
    },
    {
        "block_type": _MBConv6,
        "kernel": 3,
        "output_size": 24,
        "num_layers": 2,
        "stride": 2,
    },
    {
        "block_type": _MBConv6,
        "kernel": 5,
        "output_size": 40,
        "num_layers": 2,
        "stride": 2,
    },
    {
        "block_type": _MBConv6,
        "kernel": 3,
        "output_size": 80,
        "num_layers": 3,
        "stride": 2,
    },
    {
        "block_type": _MBConv6,
        "kernel": 5,
        "output_size": 112,
        "num_layers": 3,
        "stride": 1,
    },
    {
        "block_type": _MBConv6,
        "kernel": 5,
        "output_size": 192,
        "num_layers": 4,
        "stride": 2,
    },
    {
        "block_type": _MBConv6,
        "kernel": 3,
        "output_size": 320,
        "num_layers": 1,
        "stride": 1,
    },
]


def checkPatchDimensions(patch_size, numlay):
    """
    Check if patch dimensions are divisible by 2^numlay.

    Args:
        patch_size (tuple or int): Tuple of integers or a single integer representing the patch size.
        numlay (int): Number of layers.

    Returns:
        int: The largest number of layers that are divisible by 2^numlay.
    """
    patch_size_to_check = np.atleast_1d(patch_size)  # Convert to array if not already
    if patch_size_to_check[-1] == 1:
        patch_size_to_check = patch_size_to_check[:-1]

    if all(patch_size_to_check >= 2**numlay) and all(
        patch_size_to_check % 2**numlay == 0
    ):
        return numlay
    else:
        base2 = np.array(
            [np.log2(x).astype(int) for x in patch_size_to_check]
        )  # get largest possible number of layers for each dim
        return int(np.min(base2))


def num_channels(default_chan, width_factor, divisor):
    """
    Compute the number of channels closest to default_chan * width_factor that is divisible by divisor.

    Args:
        default_chan (int): Default number of channels.
        width_factor (float): Width factor.
        divisor (int): Divisor.

    Returns:
        int: The number of channels closest to default_chan * width_factor that is divisible by divisor.
    """
    default_chan *= width_factor
    new_out = int(default_chan + divisor / 2) // divisor * divisor
    new_out = max(new_out, divisor)

    if new_out < 0.9 * default_chan:
        new_out += divisor

    return int(new_out)


def num_layers(default_lay, depth_factor):
    """
    Compute the number of layers closest to default_lay * depth_factor.

    Args:
        default_lay (int): Default number of layers.
        depth_factor (float): Depth factor.

    Returns:
        int: The number of layers closest to default_lay * depth_factor.
    """
    return int(math.ceil(default_lay * depth_factor))


class EfficientNet(ModelBase):
    """
    EfficientNet model with customizable scaling of depth and width.

    Args:
        parameters (dict): A dictionary of configuration parameters for the model.
        scale_params (dict) - A dictionary defining scaling of depth and width for the model.
    """

    def __init__(self, parameters: dict, scale_params):  # how to scale depth and width
        super(EfficientNet, self).__init__(parameters)

        # check/define defaults

        if self.n_dimensions == 2:
            self.output_size = (1, 1)
        elif self.n_dimensions == 3:
            self.output_size = (1, 1, 1)
        if self.Norm is None:
            sys.stderr.write(
                "Warning: efficientnet is not defined without a normalization layer"
            )
            self.Norm = self.BatchNorm

        patch_check = checkPatchDimensions(parameters["patch_size"], numlay=5)
        self.DEFAULT_BLOCKS = DEFAULT_BLOCKS

        common_msg = "The patch size is not large enough for desired number of layers. It is expected that each dimension of the patch size is divisible by 2^i, where i is in a integer greater than or equal to 2."

        assert not (patch_check != 5 and patch_check <= 1), common_msg

        if patch_check != 5 and patch_check >= 2:
            downsamp = np.where(
                np.array([x["stride"] == 2 for x in self.DEFAULT_BLOCKS])
            )[0][patch_check - 1]
            self.DEFAULT_BLOCKS = self.DEFAULT_BLOCKS[:downsamp]
            print(common_msg + " Only the first %d layers will run." % patch_check)

        num_out_channels = num_channels(32, scale_params["width"], 8)
        # first convolution: 3x3 conv stride 2, norm, swish
        self.features = [
            (
                "conv1",
                self.Conv(
                    in_channels=self.n_channels,
                    out_channels=num_out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            ),
            ("norm1", self.Norm(num_out_channels)),
            ("swish1", nn.SiLU(inplace=True)),
        ]
        self.features = nn.Sequential(OrderedDict(self.features))

        for i, block in enumerate(self.DEFAULT_BLOCKS):
            for i_lay in range(
                0, num_layers(block["num_layers"], scale_params["depth"])
            ):
                temp_out = num_channels(block["output_size"], scale_params["width"], 8)
                layer = block["block_type"](
                    num_in_feats=num_out_channels,
                    num_out_feats=temp_out,
                    kernel_size=block["kernel"],
                    Norm=self.Norm,
                    Conv=self.Conv,
                    Pool=self.AdaptiveAvgPool,
                    stride=(block["stride"] - 1) * (i_lay == 0) + 1,
                    output_size=self.output_size,
                    reduction=4,
                )
                self.features.add_module("block%d-layer%d" % (i, i_lay), layer)
                num_out_channels = temp_out

        # final convolution : conv 1x1, pooling, fc

        final_conv = [
            (
                "conv2",
                self.Conv(
                    in_channels=num_out_channels,
                    out_channels=num_channels(1280, scale_params["width"], 8),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            ),
            ("norm2", self.Norm(num_channels(1280, scale_params["width"], 8))),
            ("swish2", nn.SiLU(inplace=True)),
        ]

        self.features.add_module("final_conv", nn.Sequential(OrderedDict(final_conv)))

        self.classifier = nn.Sequential(
            nn.Dropout(p=scale_params["dropout"], inplace=True),
            nn.Linear(num_channels(1280, scale_params["width"], 8), self.n_classes),
        )

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
        Applies the model's layers to the input tensor and produces an output tensor.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: Output tensor produced by the model.
        """
        # Pass the input tensor through the feature extraction layers
        features = self.features(x)

        # Apply adaptive average pooling to the output tensor from the previous step
        out = self.AdaptiveAvgPool(self.output_size)(features).view(
            features.size(0), -1
        )

        # Pass the tensor through the classifier layers
        out = self.classifier(out)

        # Apply the final convolution layer if not None
        if not self.final_convolution_layer is None:
            if self.final_convolution_layer == F.softmax:
                out = self.final_convolution_layer(out, dim=1)
            else:
                out = self.final_convolution_layer(out)

        return out


def efficientnetB0(parameters):
    return EfficientNet(
        parameters, scale_params={"width": 1.0, "depth": 1.0, "dropout": 0.2}
    )


def efficientnetB1(parameters):
    return EfficientNet(
        parameters, scale_params={"width": 1.0, "depth": 1.1, "dropout": 0.2}
    )


def efficientnetB2(parameters):
    return EfficientNet(
        parameters, scale_params={"width": 1.1, "depth": 1.2, "dropout": 0.3}
    )


def efficientnetB3(parameters):
    return EfficientNet(
        parameters, scale_params={"width": 1.2, "depth": 1.4, "dropout": 0.3}
    )


def efficientnetB4(parameters):
    return EfficientNet(
        parameters, scale_params={"width": 1.4, "depth": 1.8, "dropout": 0.4}
    )


def efficientnetB5(parameters):
    return EfficientNet(
        parameters, scale_params={"width": 1.6, "depth": 2.2, "dropout": 0.4}
    )


def efficientnetB6(parameters):
    return EfficientNet(
        parameters, scale_params={"width": 1.8, "depth": 2.6, "dropout": 0.5}
    )


def efficientnetB7(parameters):
    return EfficientNet(
        parameters, scale_params={"width": 2.0, "depth": 3.1, "dropout": 0.5}
    )
