import torch
import torch.nn as nn
import torch.nn.functional as F


class out_conv(nn.Module):
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
    ):
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

        self.conv0 = Conv(input_channels, input_channels // 2, **conv_kwargs)
        self.conv1 = Conv(input_channels // 2, input_channels // 2, **conv_kwargs)
        self.conv2 = Conv(input_channels // 2, input_channels // 2, **conv_kwargs)
        self.conv3 = Conv(input_channels // 2, output_channels, **conv_kwargs)

        self.in_0 = (
            norm(input_channels, **norm_kwargs) if norm is not None else nn.Identity()
        )
        self.in_1 = (
            norm(input_channels//2, **norm_kwargs) if norm is not None else nn.Identity()
        )
        self.in_2 = (
            norm(input_channels//2, **norm_kwargs) if norm is not None else nn.Identity()
        )
        self.in_3 = (
            norm(input_channels//2, **norm_kwargs) if norm is not None else nn.Identity()
        )

        self.act = self.act(**act_kwargs)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.act(self.in_0(x))

        x = self.conv0(x)
        if self.res == True:
            skip = x
        x = self.act(self.in_1(x))

        x = self.act(self.in_2(self.conv1(x)))
        x = self.conv2(x)
        if self.res == True:
            x = x + skip

        x = self.act(self.in_3(x))
        x = self.conv3(x)

        if not (self.final_convolution_layer == None):
            if self.final_convolution_layer == F.softmax:
                x = self.final_convolution_layer(x, dim=1)
            else:
                x = self.final_convolution_layer(x)
        return x
