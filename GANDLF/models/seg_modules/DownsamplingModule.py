import torch.nn as nn


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
        """
        Initializes a downsampling module with convolution, batch normalization, and activation layers.

        Args:
            input_channels (int): Number of input channels
            output_channels (int): Number of output channels
            conv (nn.Module): Convolution layer
            conv_kwargs (dict): Convolution layer keyword arguments
            norm (nn.Module): Normalization layer
            norm_kwargs (dict): Normalization layer keyword arguments
            act (nn.Module): Activation function layer
            act_kwargs (dict): Activation function layer keyword arguments
        """
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

        self.in_0 = norm(input_channels, **norm_kwargs)

        self.conv0 = conv(input_channels, output_channels, **conv_kwargs)

        self.act = act(**act_kwargs)

    def forward(self, x):
        """
        Applies a downsampling operation to the input tensor.

        [input --> in --> lrelu --> ConvDS --> output]

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: The output tensor, of shape (batch_size, output_channels, height // 2, width // 2).
        """
        x = self.conv0(self.act(self.in_0(x)))

        return x
