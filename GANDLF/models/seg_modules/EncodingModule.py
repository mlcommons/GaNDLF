import torch.nn as nn


class EncodingModule(nn.Module):
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
        dropout=nn.Dropout2d,
        dropout_kwargs=None,
        network_kwargs=None,
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
        if dropout_kwargs is None:
            dropout_kwargs = {"p": 0.5, "inplace": False}
        if network_kwargs is None:
            network_kwargs = {"res": False}

        self.residual = network_kwargs["res"]

        self.conv0 = conv(output_channels, output_channels, **conv_kwargs)
        self.conv1 = conv(output_channels, output_channels, **conv_kwargs)

        self.in_0 = norm(output_channels, **norm_kwargs)
        self.in_1 = norm(output_channels, **norm_kwargs)

        self.act = act(**act_kwargs)

        self.dropout = dropout(**dropout_kwargs)

    def forward(self, x):
        """
        The forward function for initial convolution.

        [input --> | --> in --> lrelu --> conv0 --> dropout --> in -|
                   |                                                |
        output <-- + <-------------------------- conv1 <-- lrelu <--|]

        Arguments:
            x {[Tensor]} -- [Takes in a type of torch Tensor]

        Returns:
            [Tensor] -- [Returns a torch Tensor]
        """
        if self.residual:
            skip = x
        x = self.act(self.in_0(x))

        x = self.conv0(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.act(self.in_1(x))

        x = self.conv1(x)
        if self.residual:
            x = x + skip

        return x
