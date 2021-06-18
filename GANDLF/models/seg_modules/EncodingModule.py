import torch
import torch.nn as nn
import torch.nn.functional as F


class EncodingModule(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        Conv,
        Dropout,
        Norm,
        kernel_size=3,
        dropout_p=0.3,
        leakiness=1e-2,
        conv_bias=True,
        norm_affine=True,
        res=False,
        lrelu_inplace=True,
    ):
        """[The Encoding convolution module to learn the information and use later]

        [This function will create the Learning convolutions]

        Arguments:
            input_channels {[int]} -- [the input number of channels, in our case
                                       the number of channels from downsample]
            output_channels {[int]} -- [the output number of channels, will det-
                                        -ermine the upcoming channels]

        Keyword Arguments:
            kernel_size {number} -- [size of filter] (default: {3})
            dropout_p {number} -- [dropout probability] (default: {0.3})
            leakiness {number} -- [the negative leakiness] (default: {1e-2})
            conv_bias {bool} -- [to use the bias in filters] (default: {True})
            norm_affine {bool} -- [affine use in norm] (default: {True})
            res {bool} -- [to use residual connections] (default: {False})
            lrelu_inplace {bool} -- [To update conv outputs with lrelu outputs]
                                    (default: {True})
        """
        nn.Module.__init__(self)
        self.res = res
        self.dropout_p = dropout_p
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.norm_affine = norm_affine
        self.lrelu_inplace = lrelu_inplace
        self.dropout = nn.Dropout3d(dropout_p)
        self.in_0 = (
            Norm(output_channels, affine=self.norm_affine, track_running_stats=True)
            if Norm is not None
            else nn.Identity()
        )
        self.in_1 = (
            Norm(output_channels, affine=self.norm_affine, track_running_stats=True)
            if Norm is not None
            else nn.Identity()
        )
        self.conv0 = Conv(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=self.conv_bias,
        )
        self.conv1 = Conv(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=self.conv_bias,
        )

    def forward(self, x):
        """The forward function for initial convolution

        [input --> | --> in --> lrelu --> conv0 --> dropout --> in -|
                   |                                                |
        output <-- + <-------------------------- conv1 <-- lrelu <--|]

        Arguments:
            x {[Tensor]} -- [Takes in a type of torch Tensor]

        Returns:
            [Tensor] -- [Returns a torch Tensor]
        """
        if self.res == True:
            skip = x
        x = F.leaky_relu(
            self.in_0(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace
        )
        x = self.conv0(x)
        if self.dropout_p is not None and self.dropout_p > 0:
            x = self.dropout(x)
        x = F.leaky_relu(
            self.in_1(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace
        )
        x = self.conv1(x)
        if self.res == True:
            x = x + skip
        return x
