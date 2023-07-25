import torch
import torch.nn as nn


class DecodingModule(nn.Module):
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
    ):
        """
        Decoding Module

        Args:
            input_channels (int): Number of input channels
            output_channels (int): Number of output channels
            conv (nn.Module): Convolution layer
            conv_kwargs (dict): Convolution layer keyword arguments
            norm (nn.Module): Normalization layer
            norm_kwargs (dict): Normalization layer keyword arguments
            act (nn.Module): Activation function layer
            act_kwargs (dict): Activation function layer keyword arguments
            network_kwargs (dict): Network keyword arguments

        """
        nn.Module.__init__(self)
        # Dev note : This should have been a super
        # super(DecodingModule, self).__init__()
        # but need to test it more

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

        self.conv0 = conv(input_channels, output_channels, **conv_kwargs)
        self.conv1 = conv(output_channels, output_channels, **conv_kwargs)
        self.conv2 = conv(output_channels, output_channels, **conv_kwargs)

        self.in_0 = norm(input_channels, **norm_kwargs)
        self.in_1 = norm(output_channels, **norm_kwargs)
        self.in_2 = norm(output_channels, **norm_kwargs)

        self.act = act(**act_kwargs)

    def forward(self, x1, x2):
        """
        Forward pass of the decoding module

        Args:
            x1 (torch.Tensor): First input tensor
            x2 (torch.Tensor): Second input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = torch.cat([x1, x2], dim=1)
        x = self.conv0(self.act(self.in_0(x)))

        if self.residual:
            skip = x

        x = self.conv1(self.act(self.in_1(x)))
        x = self.conv2(self.act(self.in_2(x)))

        if self.residual:
            x = x + skip

        return x
