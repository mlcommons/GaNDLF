import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
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
        self.residual = res
        self.output_channels = output_channels
        self.dropout_p = dropout_p
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.inst_norm_affine = inst_norm_affine
        self.lrelu_inplace = lrelu_inplace
        self.dropout = Dropout(dropout_p)
        self.inst_norm = InstanceNorm(
            int(output_channels / 4),
            affine=self.inst_norm_affine,
            track_running_stats=True,
        )
        self.inst_norm_final = InstanceNorm(
            output_channels, affine=self.inst_norm_affine, track_running_stats=True
        )
        self.conv_1x1 = Conv(
            output_channels,
            int(output_channels / 4),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.conv_bias,
        )
        self.conv_3x3 = Conv(
            int(output_channels / 4),
            int(output_channels / 4),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.conv_bias,
        )
        self.conv_1x1_final = Conv(
            output_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.conv_bias,
        )

    def forward(self, x):
        if self.residual:
            skip = x
        x1 = F.leaky_relu(
            self.inst_norm(self.conv_1x1(x)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )

        x2 = F.leaky_relu(
            self.inst_norm(self.conv_1x1(x)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )
        x2 = F.leaky_relu(
            self.inst_norm(self.conv_3x3(x2)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )

        x3 = F.leaky_relu(
            self.inst_norm(self.conv_1x1(x)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )
        x3 = F.leaky_relu(
            self.inst_norm(self.conv_3x3(x3)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )
        x3 = F.leaky_relu(
            self.inst_norm(self.conv_3x3(x3)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )

        x4 = F.leaky_relu(
            self.inst_norm(self.conv_1x1(x)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )
        x4 = F.leaky_relu(
            self.inst_norm(self.conv_3x3(x4)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )
        x4 = F.leaky_relu(
            self.inst_norm(self.conv_3x3(x4)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )
        x4 = F.leaky_relu(
            self.inst_norm(self.conv_3x3(x4)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.inst_norm_final(self.conv_1x1_final(x))

        x = x + skip
        x = F.leaky_relu(x, negative_slope=self.leakiness, inplace=self.lrelu_inplace)

        return x
