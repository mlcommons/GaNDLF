import torch
import torch.nn as nn
import torch.nn.functional as F
from .Interpolate import Interpolate


class FCNUpsamplingModule(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        Conv,
        leakiness=1e-2,
        lrelu_inplace=True,
        kernel_size=3,
        scale_factor=2,
        conv_bias=True,
        inst_norm_affine=True,
    ):
        """[summary]

        [description]

        Arguments:
            input__channels {[type]} -- [description]
            output_channels {[type]} -- [description]

        Keyword Arguments:
            leakiness {number} -- [description] (default: {1e-2})
            lrelu_inplace {bool} -- [description] (default: {True})
            kernel_size {number} -- [description] (default: {3})
            scale_factor {number} -- [description] (default: {2})
            conv_bias {bool} -- [description] (default: {True})
            inst_norm_affine {bool} -- [description] (default: {True})
        """
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.scale_factor = scale_factor
        if Conv == nn.Conv3d:
            mode = "trilinear"
        else:
            mode = "bilinear"
        self.interpolate = Interpolate(
            scale_factor=2 ** (self.scale_factor - 1), mode=mode, align_corners=True
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
        # print("Pre interpolate and conv:", x.shape)
        x = self.interpolate(self.conv0(x))
        # print("Post interpolate and conv:", x.shape)
        return x
