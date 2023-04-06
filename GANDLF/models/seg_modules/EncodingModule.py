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
        """
        Encoding module for the UNet model.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            conv (nn.Module): Convolution layer.
            conv_kwargs (dict): Convolution layer keyword arguments.
            norm (nn.Module): Normalization layer.
            norm_kwargs (dict): Normalization layer keyword arguments.
            act (nn.Module): Activation function layer.
            act_kwargs (dict): Activation function layer keyword arguments.
            dropout (nn.Module): Dropout layer.
            dropout_kwargs (dict): Dropout layer keyword arguments.
            residual (bool): Flag for using residual connection.
        """
        nn.Module.__init__(self)
        # Dev note : This should have been a super
        # super(EncodingModule, self).__init__()
        # but need to test it more

        # Set default arguments
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

        # Set attributes together
        self.residual = network_kwargs["res"]
        self.act = act(**act_kwargs)
        self.dropout = dropout(**dropout_kwargs) if dropout is not None else None

        # Define layers together
        self.in_0 = norm(output_channels, **norm_kwargs)
        self.in_1 = norm(output_channels, **norm_kwargs)

        self.conv0 = conv(input_channels, output_channels, **conv_kwargs)
        self.conv1 = conv(output_channels, output_channels, **conv_kwargs)

    def forward(self, x):
        """
        The forward function for encoding module.

        [input --> | --> in --> lrelu --> conv0 --> dropout --> in -|
                   |                                                |
        output <-- + <-------------------------- conv1 <-- lrelu <--|]

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
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
