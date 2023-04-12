import torch.nn as nn
import torch.nn.functional as F


class IncDownsamplingModule(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        Conv,
        Dropout,
        InstanceNorm,
        leakiness=1e-2,
        kernel_size=1,
        conv_bias=True,
        inst_norm_affine=True,
        lrelu_inplace=True,
    ):
        """
        A module that performs downsampling using a 1x1 convolution followed by a stride of 2.
        The output tensor is then passed through an instance normalization layer and a leaky ReLU activation function.

        Args:
            input_channels (int): The number of input channels in the tensor.
            output_channels (int): The number of output channels in the tensor.
            Conv (torch.nn.Module): The type of convolution layer to use.
            Dropout (torch.nn.Module): The type of dropout layer to use.
            InstanceNorm (torch.nn.Module): The type of instance normalization layer to use.
            leakiness (float, optional): The negative slope of the leaky ReLU activation function. Defaults to 1e-2.
            kernel_size (int, optional): The kernel size of the 1x1 convolution layer. Defaults to 1.
            conv_bias (bool, optional): Whether to include a bias term in the 1x1 convolution layer.
                                        Defaults to True.
            inst_norm_affine (bool, optional): Whether to include an affine transform in the instance normalization layer.
                                               Defaults to True.
            lrelu_inplace (bool, optional): Whether to use the input tensor for the output of the leaky ReLU activation function.
                                            Defaults to True.
        """

        nn.Module.__init__(self)

        self.leakiness = leakiness
        self.lrelu_inplace = lrelu_inplace
        # Instantiate instance normalization and 1x1 convolution layers
        self.inst_norm = InstanceNorm(
            output_channels, affine=inst_norm_affine, track_running_stats=True
        )

        # Instantiate 1x1 convolution layer
        # Dev Note: This currently makes little sense if kernel size is 1
        # and stride is 2 as information is lost with this
        self.down = Conv(
            input_channels,
            output_channels,
            kernel_size=1,
            stride=2,
            padding=0,
            bias=conv_bias,
        )

    def forward(self, x):
        """
        Performs a forward pass through the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Pass input tensor through 1x1 convolution, instance normalization, and leaky ReLU activation function
        x = F.leaky_relu(
            self.inst_norm(self.down(x)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )
        return x
