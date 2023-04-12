import torch
import torch.nn as nn
import torch.nn.functional as F


class out_conv(nn.Module):
    """
    A class for the output convolution block in a neural network, which includes a convolutional layer,
    batch normalization layer, activation function, and final convolutional layer (e.g. sigmoid).
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        conv=nn.Conv2d,
        conv_kwargs=None,
        norm=nn.BatchNorm2d,
        norm_kwargs=None,
        act=nn.LeakyReLU,
        act_kwargs=None,
        network_kwargs=None,
        final_convolution_layer=nn.Sigmoid,
        sigmoid_input_multiplier=1.0,
    ):
        """
        Args:
            input_channels (int): number of input channels.
            output_channels (int): number of output channels.
            conv (torch.nn.Module): convolutional layer to use (default: nn.Conv2d).
            conv_kwargs (dict): keyword arguments to pass to the convolutional layer (default: None).
            norm (torch.nn.Module): normalization layer to use (default: nn.BatchNorm2d).
            norm_kwargs (dict): keyword arguments to pass to the normalization layer (default: None).
            act (torch.nn.Module): activation function to use (default: nn.LeakyReLU).
            act_kwargs (dict): keyword arguments to pass to the activation function (default: None).
            network_kwargs (dict): keyword arguments to pass to the network (default: {"res": False}).
            final_convolution_layer (torch.nn.Module): final convolutional layer to use (default: nn.Sigmoid).
            sigmoid_input_multiplier (float): multiplier for input to sigmoid function (default: 1.0).
        """
        nn.Module.__init__(self)
        if conv_kwargs is None:
            conv_kwargs = {"kernel_size": 3, "stride": 1, "padding": 1, "bias": True}
        if norm_kwargs is None:
            norm_kwargs = {
                "eps": 1e-5,
                "affine": True,
                "momentum": 0.1,
                "track_running_stats": True,
            }
        if act_kwargs is None:
            act_kwargs = {"negative_slope": 1e-2, "inplace": True}
        if network_kwargs is None:
            network_kwargs = {"res": False}

        self.residual = network_kwargs["res"]
        self.sigmoid_input_multiplier = sigmoid_input_multiplier

        self.conv0 = conv(input_channels, output_channels, **conv_kwargs)

        self.in_0 = (
            norm(input_channels, **norm_kwargs) if norm is not None else nn.Identity()
        )

        self.act = act(**act_kwargs)

        self.final_convolution_layer = final_convolution_layer

    def forward(self, x):
        """
        Perform forward pass with the given input.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            x (torch.Tensor): output tensor after passing through the output convolution block.
        """
        x = self.conv0(self.act(self.in_0(x)))

        if not (self.final_convolution_layer is None):
            if self.final_convolution_layer == F.softmax:
                x = self.final_convolution_layer(x, dim=1)
            elif self.final_convolution_layer == torch.sigmoid:
                x = torch.sigmoid(self.sigmoid_input_multiplier * x)
            else:
                x = self.final_convolution_layer(x)
        return x
