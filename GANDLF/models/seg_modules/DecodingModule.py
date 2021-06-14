import torch
import torch.nn as nn
import torch.nn.functional as F


class DecodingModule(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        Conv,
        Dropout,
        InstanceNorm,
        leakiness=1e-2,
        conv_bias=True,
        kernel_size=3,
        inst_norm_affine=True,
        res=True,
        lrelu_inplace=True,
    ):
        """[The Decoding convolution module to learn the information and use later]

        [This function will create the Learning convolutions]

        Arguments:
            input_channels {[int]} -- [the input number of channels, in our case
                                       the number of channels from downsample]
            output_channels {[int]} -- [the output number of channels, will det-
                                        -ermine the upcoming channels]

        Keyword Arguments:
            kernel_size {number} -- [size of filter] (default: {3})
            leakiness {number} -- [the negative leakiness] (default: {1e-2})
            conv_bias {bool} -- [to use the bias in filters] (default: {True})
            inst_norm_affine {bool} -- [affine use in norm] (default: {True})
            res {bool} -- [to use residual connections] (default: {False})
            lrelu_inplace {bool} -- [To update conv outputs with lrelu outputs]
                                    (default: {True})
        """
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.res = res
        self.in_0 = InstanceNorm(
            input_channels, affine=self.inst_norm_affine, track_running_stats=True
        )
        self.in_1 = InstanceNorm(
            output_channels, affine=self.inst_norm_affine, track_running_stats=True
        )
        self.in_2 = InstanceNorm(
            output_channels, affine=self.inst_norm_affine, track_running_stats=True
        )
        self.conv0 = Conv(
            input_channels,
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
        self.conv2 = Conv(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=self.conv_bias,
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        # print(x.shape)
        x = F.leaky_relu(self.in_0(x))
        x = self.conv0(x)
        if self.res == True:
            skip = x
        x = F.leaky_relu(self.in_1(x))
        x = F.leaky_relu(self.in_2(self.conv1(x)))
        x = self.conv2(x)
        if self.res == True:
            x = x + skip
        return x
