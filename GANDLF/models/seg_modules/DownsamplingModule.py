import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsamplingModule(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        Conv,
        Dropout,
        InstanceNorm,
        leakiness=1e-2,
        dropout_p=0.3,
        kernel_size=3,
        conv_bias=True,
        inst_norm_affine=True,
        lrelu_inplace=True,
    ):
        """[To Downsample a given input with convolution operation]

        [This one will be used to downsample a given comvolution while doubling
        the number filters]

        Arguments:
            input_channels {[int]} -- [The input number of channels are taken
                                       and then are downsampled to double usually]
            output_channels {[int]} -- [the output number of channels are
                                        usually the double of what of input]

        Keyword Arguments:
            leakiness {float} -- [the negative leakiness] (default: {1e-2})
            conv_bias {bool} -- [to use the bias in filters] (default: {True})
            inst_norm_affine {bool} -- [affine use in norm] (default: {True})
            lrelu_inplace {bool} -- [To update conv outputs with lrelu outputs]
                                    (default: {True})
        """
        # nn.Module.__init__(self)
        super(DownsamplingModule, self).__init__()
        self.dropout_p = dropout_p
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.inst_norm_affine = inst_norm_affine
        self.lrelu_inplace = True
        self.in_0 = InstanceNorm(
            output_channels, affine=self.inst_norm_affine, track_running_stats=True
        )
        self.conv0 = Conv(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=2,
            padding=(kernel_size - 1) // 2,
            bias=self.conv_bias,
        )

    def forward(self, x):
        """[This is a forward function for ]

        [input -- > in --> lrelu --> ConvDS --> output]

        Arguments:
            x {[Tensor]} -- [Takes in a type of torch Tensor]

        Returns:
            [Tensor] -- [Returns a torch Tensor]
        """
        x = F.leaky_relu(
            self.in_0(self.conv0(x)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )
        # print(x.shape)
        return x
