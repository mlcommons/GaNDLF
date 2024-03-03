import torch.nn as nn


class IncDropout(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        Conv: nn.Module = nn.Conv2d,
        Dropout: nn.Module = nn.Dropout2d,
        InstanceNorm: nn.Module = nn.InstanceNorm2d,
        dropout_p: float = 0.3,
        leakiness: float = 1e-2,
        conv_bias: bool = True,
        inst_norm_affine: bool = True,
        res: bool = False,
        lrelu_inplace: bool = True,
    ):
        """
        Incremental dropout module.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            Conv (nn.Module, optional): The convolutional layer type. Defaults to nn.Conv2d.
            Dropout (nn.Module, optional): The dropout layer type. Defaults to nn.Dropout2d.
            InstanceNorm (nn.Module, optional): The instance normalization layer type. Defaults to nn.InstanceNorm2d.
            dropout_p (float, optional): The dropout probability. Defaults to 0.3.
            leakiness (float, optional): The leakiness of the leaky ReLU. Defaults to 1e-2.
            conv_bias (bool, optional): The bias in the convolutional layer. Defaults to True.
            inst_norm_affine (bool, optional): Whether to use the affine transformation in the instance normalization layer. Defaults to True.
            res (bool, optional): Whether to use residual connections. Defaults to False.
            lrelu_inplace (bool, optional): Whether to use the inplace version of the leaky ReLU. Defaults to True.
        """
        nn.Module.__init__(self)

        self.dropout = Dropout(dropout_p)
        self.conv = Conv(
            input_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=conv_bias,
        )

    def forward(self, x):
        """
        Forward pass of the incremental dropout module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            x (torch.Tensor): The output tensor.
        """
        x = self.dropout(x)
        x = self.conv(x)
        return x
