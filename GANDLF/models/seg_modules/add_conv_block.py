# -*- coding: utf-8 -*-
"""
Add Conv Block for MSDNet
"""


def add_conv_block(
    Conv,
    BatchNorm,
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    dilation=1,
    last=False,
):
    """
    Adds a convolution block to a neural network. A convolution block consists of a convolutional layer and a batch
    normalization layer.

    Args:
        Conv (class): Convolutional layer class.
        BatchNorm (class): Batch normalization layer class.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size of the convolutional layer. Default is 3.
        dilation (int, optional): Dilation rate of the convolutional layer. Default is 1.
        last (bool, optional): Indicates if this is the last block in the neural network. Default is False.

    Returns:
        list :A list containing the convolutional layer and batch normalization layer.

    """
    padding = dilation if not last else 0
    conv_layer = Conv(
        in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
    )
    bn_layer = BatchNorm(out_channels)

    return [conv_layer, bn_layer]
