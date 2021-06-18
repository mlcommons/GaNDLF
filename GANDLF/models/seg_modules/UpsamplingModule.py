import torch
import torch.nn as nn
import torch.nn.functional as F
from GANDLF.models.seg_modules.Interpolate import Interpolate


class UpsamplingModule(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        Conv,
        kernel_size=3,
        scale_factor=2,
        conv_bias=True,
    ):
        """[summary]

        [description]

        Arguments:
            input__channels {[type]} -- [description]
            output_channels {[type]} -- [description]

        Keyword Arguments:
            kernel_size {number} -- [description] (default: {3})
            scale_factor {number} -- [description] (default: {2})
            conv_bias {bool} -- [description] (default: {True})
        """
        nn.Module.__init__(self)
        self.conv_bias = conv_bias
        self.scale_factor = scale_factor
        if Conv == nn.Conv3d:
            mode = "trilinear"
        else:
            mode = "bilinear"
        self.interpolate = Interpolate(
            scale_factor=self.scale_factor, mode=mode, align_corners=True
        )
        self.conv0 = Conv(
            input_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.conv_bias,
        )

    def forward(self, x):
        """[summary]

        [description]

        Extends:
        """
        x = self.conv0(self.interpolate(x))
        return x
