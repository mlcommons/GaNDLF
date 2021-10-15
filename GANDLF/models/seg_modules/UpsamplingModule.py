import torch.nn as nn
from GANDLF.models.seg_modules.Interpolate import Interpolate


class UpsamplingModule(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        conv=nn.Conv2d,
        conv_kwargs=None,
        scale_factor=2,
        interpolation_mode = "trilinear",
    ):
        nn.Module.__init__(self)
        if conv_kwargs is None:
            conv_kwargs = {"kernel_size": 1, "stride": 1, "padding": 0, "bias": True}

        self.interp_kwargs = {
            "size": None,
            "scale_factor": scale_factor,
            "mode": interpolation_mode,
            "align_corners": True,
        }
        self.interpolate = Interpolate(interp_kwargs=self.interp_kwargs)

        self.conv0 = conv(input_channels, output_channels, **conv_kwargs)

    def forward(self, x):
        x = self.conv0(self.interpolate(x))
        return x
