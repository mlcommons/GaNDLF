import torch.nn as nn
import torch.nn.functional as F


class IncUpsamplingModule(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        ConvTranspose,
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
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dropout_p = dropout_p
        self.leakiness = leakiness
        self.conv_bias = conv_bias
        self.inst_norm_affine = inst_norm_affine
        self.residual = res
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm = InstanceNorm(
            output_channels, affine=self.inst_norm_affine, track_running_stats=True
        )
        self.up = ConvTranspose(
            input_channels,
            output_channels,
            kernel_size=1,
            stride=2,
            padding=0,
            output_padding=1,
            bias=self.conv_bias,
        )

    def forward(self, x):
        x = F.leaky_relu(
            self.inst_norm(self.up(x)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )
        return x
