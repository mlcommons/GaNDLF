import torch.nn as nn
import torch.nn.functional as F


class DownsamplingModule(nn.Module):
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
    ):
        super(DownsamplingModule, self).__init__()
        if conv_kwargs is None:
            conv_kwargs = {"kernel_size": 3, "stride": 2, "padding": 1, "bias": True}
        if norm_kwargs is None:
            norm_kwargs = {
                "eps": 1e-5,
                "affine": True,
                "momentum": 0.1,
                "track_running_stats": True,
            }
        if act_kwargs is None:
            act_kwargs = {"negative_slope": 1e-2, "inplace": True}

        self.in_0 = norm(output_channels, **norm_kwargs)

        self.conv0 = conv(input_channels, output_channels, **conv_kwargs)

        self.act = act(**act_kwargs)

    def forward(self, x):
        """
        This is a forward function for the current module.

        [input -- > in --> lrelu --> ConvDS --> output]

        Arguments:
            x {[Tensor]} -- [Takes in a type of torch Tensor]

        Returns:
            [Tensor] -- [Returns a torch Tensor]
        """
        x = self.act(self.in_0(self.conv0(x)))

        return x
