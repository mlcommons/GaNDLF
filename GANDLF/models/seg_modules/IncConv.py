import torch.nn as nn
import torch.nn.functional as F


class IncConv(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        Conv,
        Dropout,
        InstanceNorm,
        dropout_p=0.3,
        leakiness=1e-2,
        conv_bias=True,
        inst_norm_affine=True,
        res=False,
        lrelu_inplace=True,
    ):
        """
        Constructor for the IncConv module.

        Args:
            input_channels (int): The number of input channels.
            output_channels (int): The number of output channels.
            Conv (nn.Module): The convolution module to use.
            Dropout (nn.Module): The dropout module to use.
            InstanceNorm (nn.Module): The instance normalization module to use.
            dropout_p (float, optional): The probability of dropping out a channel. Defaults to 0.3.
            leakiness (float, optional): The negative slope of the LeakyReLU activation function. Defaults to 1e-2.
            conv_bias (bool, optional): Whether to include a bias term in the convolution. Defaults to True.
            inst_norm_affine (bool, optional): Whether to include an affine transformation in the instance normalization. Defaults to True.
            res (bool, optional): Whether to use residual connections. Defaults to False.
            lrelu_inplace (bool, optional): Whether to perform the LeakyReLU activation function in-place. Defaults to True.
        """
        nn.Module.__init__(self)
        self.output_channels = output_channels
        self.leakiness = leakiness
        self.conv_bias = conv_bias
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.inst_norm = InstanceNorm(
            output_channels, affine=self.inst_norm_affine, track_running_stats=True
        )
        self.conv = Conv(
            input_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.conv_bias,
        )

    def forward(self, x):
        """
        The forward function of the IncConv module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            x (torch.Tensor): The output tensor.
        """
        x = F.leaky_relu(
            self.inst_norm(self.conv(x)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )
        return x
