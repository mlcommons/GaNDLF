import torch
import torch.nn as nn
import torch.nn.functional as F
from GANDLF.models.seg_modules.Interpolate import Interpolate


class UpsamplingModule(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        conv=nn.Conv2d,
        conv_kwargs=None,
        scale_factor=2,
    ):
        nn.Module.__init__(self)
        if conv_kwargs is None:
            conv_kwargs = {"kernel_size": 3, "stride": 1, "padding": 1, "bias": True}
        self.scale_factor = scale_factor
        if conv == nn.Conv2d:
            mode = "bilinear"
        else:
            mode = "trilinear"
        self.interpolate = Interpolate(
            scale_factor=self.scale_factor, mode=mode, align_corners=True
        )
        self.conv0 = Conv(input_channels, output_channels, **conv_kwargs)

    def forward(self, x):
        x = self.conv0(self.interpolate(x))
        return x
