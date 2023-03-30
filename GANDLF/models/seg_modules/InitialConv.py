import torch.nn as nn


class InitialConv(nn.Module):
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
        The initial convolutional layer for a UNet-like network.

        Args:
            input_channels (int): The number of channels in the input tensor.
            output_channels (int): The number of channels in the output tensor.
            conv (nn.Module): The convolutional layer to use.
            conv_kwargs (dict): The arguments to pass to the convolutional layer.
            norm (nn.Module): The normalization layer to use.
            norm_kwargs (dict): The arguments to pass to the normalization layer.
            activation (nn.Module): The activation function to use.
            activation_kwargs (dict): The arguments to pass to the activation function.
            dropout (nn.Module): The dropout layer to use.
            dropout_kwargs (dict): The arguments to pass to the dropout layer.
            network_kwargs (bool): Whether to use residual connections.
        """
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

        self.conv0 = conv(input_channels, output_channels, **conv_kwargs)
        self.conv1 = conv(output_channels, output_channels, **conv_kwargs)
        self.conv2 = conv(output_channels, output_channels, **conv_kwargs)

        self.in_0 = norm(output_channels, **norm_kwargs)
        self.in_1 = norm(output_channels, **norm_kwargs)

        self.act = act(**act_kwargs)
        self.residual = network_kwargs["res"]

        if dropout is not None:
            self.dropout = dropout(**dropout_kwargs)

    def forward(self, x):
        """
        The forward function for initial convolution.

        [input --> conv0 --> | --> in --> lrelu --> conv1 --> dropout --> in -|
                             |                                                |
                  output <-- + <-------------------------- conv2 <-- lrelu <--|]

         Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv0(x)
        if self.residual:
            skip = x
        x = self.act(self.in_0(x))

        x = self.conv1(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.act(self.in_1(x))

        x = self.conv2(x)
        if self.residual:
            x = x + skip

        return x
