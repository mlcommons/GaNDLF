import torch
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
        nn.Module.__init__(self)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dropout_p = dropout_p
        self.leakiness = leakiness
        self.conv_bias = conv_bias
        self.inst_norm_affine = inst_norm_affine
        self.res = res
        self.lrelu_inplace = lrelu_inplace
        self.dropout = Dropout(dropout_p)
        self.conv = Conv(
            input_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.conv_bias,
        )

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        return x
