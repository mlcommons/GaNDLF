# -*- coding: utf-8 -*-
"""
Implementation of Fully Convolutional Network - FCN
"""

import torch.nn.functional as F
import torch
from GANDLF.models.seg_modules.DownsamplingModule import DownsamplingModule
from GANDLF.models.seg_modules.EncodingModule import EncodingModule
from GANDLF.models.seg_modules.FCNUpsamplingModule import FCNUpsamplingModule
from GANDLF.models.seg_modules.InitialConv import InitialConv
from .modelBase import ModelBase


class fcn(ModelBase):
    """
    This is the standard FCN (Fully Convolutional Network) architecture :
    https://arxiv.org/abs/1411.4038 . The Downsampling, Encoding, Decoding modules are defined in
    the seg_modules file. These smaller modules are basically defined by 2 parameters, the input
    channels (filters) and the output channels (filters), and some other hyperparameters, which
    remain constant all the modules. For more details on the smaller modules please have a look at
    the seg_modules file.
    DOI: 10.1109/TPAMI.2016.2572683
    """

    def __init__(
        self,
        parameters: dict,
    ):
        super(fcn, self).__init__(parameters)
        self.ins = InitialConv(
            input_channels=self.n_channels,
            output_channels=self.base_filters,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
        )
        self.ds_0 = DownsamplingModule(
            input_channels=self.base_filters,
            output_channels=self.base_filters * 2,
            conv=self.Conv,
            norm=self.Norm,
        )
        self.en_1 = EncodingModule(
            input_channels=self.base_filters * 2,
            output_channels=self.base_filters * 2,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
        )
        self.ds_1 = DownsamplingModule(
            input_channels=self.base_filters * 2,
            output_channels=self.base_filters * 4,
            conv=self.Conv,
            norm=self.Norm,
        )
        self.en_2 = EncodingModule(
            input_channels=self.base_filters * 4,
            output_channels=self.base_filters * 4,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
        )
        self.ds_2 = DownsamplingModule(
            input_channels=self.base_filters * 4,
            output_channels=self.base_filters * 8,
            conv=self.Conv,
            norm=self.Norm,
        )
        self.en_3 = EncodingModule(
            input_channels=self.base_filters * 8,
            output_channels=self.base_filters * 8,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
        )
        self.ds_3 = DownsamplingModule(
            input_channels=self.base_filters * 8,
            output_channels=self.base_filters * 16,
            conv=self.Conv,
            norm=self.Norm,
        )
        self.en_4 = EncodingModule(
            input_channels=self.base_filters * 16,
            output_channels=self.base_filters * 16,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
        )
        self.us_4 = FCNUpsamplingModule(
            input_channels=self.base_filters * 16,
            output_channels=1,
            conv=self.Conv,
            scale_factor=5,
            interpolation_mode=self.linear_interpolation_mode,
        )
        self.us_3 = FCNUpsamplingModule(
            input_channels=self.base_filters * 8,
            output_channels=1,
            conv=self.Conv,
            scale_factor=4,
            interpolation_mode=self.linear_interpolation_mode,
        )
        self.us_2 = FCNUpsamplingModule(
            input_channels=self.base_filters * 4,
            output_channels=1,
            conv=self.Conv,
            scale_factor=3,
            interpolation_mode=self.linear_interpolation_mode,
        )
        self.us_1 = FCNUpsamplingModule(
            input_channels=self.base_filters * 2,
            output_channels=1,
            conv=self.Conv,
            scale_factor=2,
            interpolation_mode=self.linear_interpolation_mode,
        )
        self.us_0 = FCNUpsamplingModule(
            input_channels=self.base_filters,
            output_channels=1,
            conv=self.Conv,
            scale_factor=1,
            interpolation_mode=self.linear_interpolation_mode,
        )
        self.conv_0 = self.Conv(
            in_channels=5,
            out_channels=self.n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Should be a 5D Tensor as [batch_size, channels, x_dims, y_dims, z_dims].

        Returns:
            x (Tensor): Returns a 5D Output Tensor as [batch_size, n_classes, x_dims, y_dims, z_dims].
        """
        x1 = self.ins(x)
        x2 = self.ds_0(x1)
        x2 = self.en_1(x2)
        x3 = self.ds_1(x2)
        x3 = self.en_2(x3)
        x4 = self.ds_2(x3)
        x4 = self.en_3(x4)
        x5 = self.ds_3(x4)
        x5 = self.en_4(x5)

        u5 = self.us_4(x5)
        u4 = self.us_3(x4)
        u3 = self.us_2(x3)
        u2 = self.us_1(x2)
        u1 = self.us_0(x1)
        x = torch.cat([u5, u4, u3, u2, u1], dim=1)
        x = self.conv_0(x)

        if not self.final_convolution_layer is None:
            if self.final_convolution_layer == F.softmax:
                x = self.final_convolution_layer(x, dim=1)
            else:
                x = self.final_convolution_layer(x)

        return x
