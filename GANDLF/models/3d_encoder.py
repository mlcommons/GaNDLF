# -*- coding: utf-8 -*-
"""
Implementation of UNet
"""

from GANDLF.models.seg_modules.DownsamplingModule import DownsamplingModule
from GANDLF.models.seg_modules.EncodingModule import EncodingModule
from GANDLF.models.seg_modules.DecodingModule import DecodingModule
from GANDLF.models.seg_modules.UpsamplingModule import UpsamplingModule
from GANDLF.models.seg_modules.in_conv import in_conv
from GANDLF.models.seg_modules.out_conv import out_conv
from GANDLF.models.seg_modules.GAPModule import GlobalAvgPool3d, GlobalAvgPool2d
from .modelBase import ModelBase


class nd_encoder(ModelBase):
    """
    This is the standard U-Net architecture : https://arxiv.org/pdf/1606.06650.pdf. The 'residualConnections' flag controls residual connections The Downsampling, Encoding, Decoding modules
    are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input channels (filters) and the output channels (filters),
    and some other hyperparameters, which remain constant all the modules. For more details on the smaller modules please have a look at the seg_modules file.
    """

    def __init__(
        self,
        n_dimensions,
        n_channels,
        n_classes,
        base_filters,
        final_convolution_layer,
        residualConnections=False,
    ):
        super(nd_encoder, self).__init__(
            n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer
        )
        self.ins = in_conv(
            self.n_channels,
            base_filters,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=residualConnections,
        )
        self.ds_0 = DownsamplingModule(
            base_filters, base_filters * 2, self.Conv, self.Dropout, self.InstanceNorm
        )
        self.en_1 = EncodingModule(
            base_filters * 2,
            base_filters * 2,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=residualConnections,
        )
        self.ds_1 = DownsamplingModule(
            base_filters * 2,
            base_filters * 4,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
        )
        self.en_2 = EncodingModule(
            base_filters * 4,
            base_filters * 4,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=residualConnections,
        )
        self.ds_2 = DownsamplingModule(
            base_filters * 4,
            base_filters * 8,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
        )
        self.en_3 = EncodingModule(
            base_filters * 8,
            base_filters * 8,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=residualConnections,
        )
        self.ds_3 = DownsamplingModule(
            base_filters * 8,
            base_filters * 16,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
        )
        self.en_4 = EncodingModule(
            base_filters * 16,
            base_filters * 32,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=residualConnections,
        )
        # self.ds_4 = DownsamplingModule(
        #     base_filters * 16,
        #     base_filters * 32,
        #     self.Conv,
        #     self.Dropout,
        #     self.InstanceNorm,
        # )
        # self.en_5 = EncodingModule(
        #     base_filters * 32,
        #     base_filters * 64,
        #     self.Conv,
        #     self.Dropout,
        #     self.InstanceNorm,
        #     res=residualConnections,
        # )
        self.gap = GlobalAvgPool3d() if n_channels == 3 else GlobalAvgPool2D()

        self.out = EncodingModule(
            base_filters * 64,
            n_classes,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=False,
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Should be a 5D Tensor as [batch_size, channels, x_dims, y_dims, zdims].

        Returns
        -------
        x : Tensor
            Returns a 5D Output Tensor as [batch_size, n_classes, x_dims, y_dims, zdims].

        """
        x = self.ins(x)
        x = self.ds_0(x)
        x = self.en_1(x)
        x = self.ds_1(x)
        x = self.en_2(x)
        x = self.ds_2(x)
        x = self.en_3(x)
        x = self.ds_3(x)
        x = self.en_4(x)
        # x = self.ds_4(x)
        # x = self.en_5(x)

        x = self.out(x)

        return x
