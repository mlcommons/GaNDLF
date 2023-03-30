import torch.nn as nn
import torch.nn.functional as F


class ResNetModule(nn.Module):
    """
    A residual block module for use in ResNet-style architectures.

    Args:
        input_channels (int): The number of input channels.
        output_channels (int): The number of output channels.
        Conv (nn.Module): The convolutional layer to use. Should be a torch.nn.Module.
        Dropout (nn.Module): The dropout layer to use. Should be a torch.nn.Module.
        InstanceNorm (nn.Module): The instance normalization layer to use. Should be a torch.nn.Module.
        dropout_p (float): The dropout probability.
        leakiness (float): The slope of the negative part of the LeakyReLU activation function.
        conv_bias (bool): Whether or not to use a bias term in the convolutional layer.
        inst_norm_affine (bool): Whether or not to use affine parameters in the instance normalization layer.
        res (bool): Whether or not to include a residual connection in the block.
        lrelu_inplace (bool): Whether or not to perform the operation in-place for the LeakyReLU activation function.

    """

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
        nn.Module.__init__(self)
        self.leakiness = leakiness
        self.residual = res
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm = InstanceNorm(
            output_channels, affine=inst_norm_affine, track_running_stats=True
        )
        self.conv = Conv(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=conv_bias,
        )

    def forward(self, x):
        """
        Forward pass for the ResNetModule.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            x (torch.Tensor): The output tensor.
        """

        # Apply residual connection if specified
        if self.residual:
            skip = x
        x = F.leaky_relu(
            self.inst_norm(self.conv(x)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )

        # Apply inst - conv - skip - relu
        x = self.inst_norm(self.conv(x))
        x = x + skip
        x = F.leaky_relu(x, negative_slope=self.leakiness, inplace=self.lrelu_inplace)

        return x
