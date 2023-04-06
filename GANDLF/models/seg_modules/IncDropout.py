import torch.nn as nn


class IncDropout(nn.Module):
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
        Incremental Dropout module with a 1x1 convolutional layer.

        Parameters
        ----------
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        Conv (torch.nn.Module, optional): Convolutional layer to use.
        Dropout (torch.nn.Module, optional): Dropout layer to use.
        InstanceNorm (torch.nn.Module, optional): Instance normalization layer to use.
        dropout_p (float, optional): Probability of an element to be zeroed. Default is 0.3.
        leakiness (float, optional): Negative slope coefficient for LeakyReLU activation. Default is 1e-2.
        conv_bias (bool, optional): If True, add a bias term to the convolutional layer. Default is True.
        inst_norm_affine (bool, optional): If True, learn two affine parameters per channel in the instance normalization layer. Default is True.
        res (bool, optional): If True, add a residual connection to the module. Default is False.
        lrelu_inplace (bool, optional): If True, perform the LeakyReLU operation in place. Default is True.
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
