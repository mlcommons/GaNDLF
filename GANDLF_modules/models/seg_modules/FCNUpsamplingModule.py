import torch.nn as nn
from GANDLF.models.seg_modules.Interpolate import Interpolate


class FCNUpsamplingModule(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        conv=nn.Conv2d,
        conv_kwargs=None,
        scale_factor=2,
        interpolation_mode="trilinear",
    ):
        """
        Upsampling module for the Fully Convolutional Network.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            conv (nn.Module): Convolution layer.
            conv_kwargs (dict): Convolution layer keyword arguments.
            scale_factor (int): Scale factor for interpolation.
            interpolation_mode (str): Interpolation mode for upsampling.

        """
        nn.Module.__init__(self)
        if conv_kwargs is None:
            conv_kwargs = {"kernel_size": 1, "stride": 1, "padding": 0, "bias": True}

        self.interpolate_kwargs = {
            "size": None,
            "scale_factor": 2 ** (scale_factor - 1),
            "mode": interpolation_mode,
            "align_corners": True,
        }
        self.interpolate = Interpolate(interp_kwargs=self.interpolate_kwargs)

        self.conv0 = conv(input_channels, output_channels, **conv_kwargs)

    def forward(self, x):
        """
        Upsampling the input tensor using the convolution layer and interpolation.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]

        Returns:
            x (torch.Tensor): Returns the upsampled tensor of shape [batch_size, output_channels, height*2, width*2].

        """
        x = self.interpolate(self.conv0(x))
        return x
