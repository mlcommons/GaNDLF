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
from .modelBase import ModelBase
import sys
from GANDLF.utils.generic import checkPatchDivisibility


class light_unet(ModelBase):
    """
    This is the LIGHT U-Net architecture.
    """

    def __init__(
        self,
        parameters: dict,
        residualConnections=False,
    ):
        self.network_kwargs = {"res": residualConnections}
        super(light_unet, self).__init__(parameters)

        self.network_kwargs = {"res": False}

        if not (checkPatchDivisibility(parameters["patch_size"])):
            sys.exit(
                "The patch size is not divisible by 16, which is required for",
                parameters["model"]["architecture"],
            )

        self.ins = in_conv(
            input_channels=self.n_channels,
            output_channels=self.base_filters,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
        )
        self.ds_0 = DownsamplingModule(
            input_channels=self.base_filters,
            output_channels=self.base_filters,
            conv=self.Conv,
            norm=self.Norm,
        )
        self.en_1 = EncodingModule(
            input_channels=self.base_filters,
            output_channels=self.base_filters,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.ds_1 = DownsamplingModule(
            input_channels=self.base_filters,
            output_channels=self.base_filters,
            conv=self.Conv,
            norm=self.Norm,
        )
        self.en_2 = EncodingModule(
            input_channels=self.base_filters,
            output_channels=self.base_filters,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.ds_2 = DownsamplingModule(
            input_channels=self.base_filters,
            output_channels=self.base_filters,
            conv=self.Conv,
            norm=self.Norm,
        )
        self.en_3 = EncodingModule(
            input_channels=self.base_filters,
            output_channels=self.base_filters,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.ds_3 = DownsamplingModule(
            input_channels=self.base_filters,
            output_channels=self.base_filters,
            conv=self.Conv,
            norm=self.Norm,
        )
        self.en_4 = EncodingModule(
            input_channels=self.base_filters,
            output_channels=self.base_filters,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.us_3 = UpsamplingModule(
            input_channels=self.base_filters,
            output_channels=self.base_filters,
            conv=self.Conv,
        )
        self.de_3 = DecodingModule(
            input_channels=self.base_filters * 2,
            output_channels=self.base_filters,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.us_2 = UpsamplingModule(
            input_channels=self.base_filters,
            output_channels=self.base_filters,
            conv=self.Conv,
        )
        self.de_2 = DecodingModule(
            input_channels=self.base_filters * 2,
            output_channels=self.base_filters,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.us_1 = UpsamplingModule(
            input_channels=self.base_filters,
            output_channels=self.base_filters,
            conv=self.Conv,
        )
        self.de_1 = DecodingModule(
            input_channels=self.base_filters * 2,
            output_channels=self.base_filters,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.us_0 = UpsamplingModule(
            input_channels=self.base_filters,
            output_channels=self.base_filters,
            conv=self.Conv,
        )
        self.out = out_conv(
            input_channels=self.base_filters * 2,
            output_channels=self.n_classes,
            conv=self.Conv,
            norm=self.Norm,
            final_convolution_layer=self.final_convolution_layer,
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Should be a 5D Tensor as [batch_size, channels, x_dims, y_dims, z_dims].

        Returns
        -------
        x : Tensor
            Returns a 5D Output Tensor as [batch_size, n_classes, x_dims, y_dims, z_dims].

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

        x = self.us_3(x5)
        x = self.de_3(x, x4)
        x = self.us_2(x)
        x = self.de_2(x, x3)
        x = self.us_1(x)
        x = self.de_1(x, x2)
        x = self.us_0(x)
        x = self.out(x, x1)
        return x


class light_resunet(light_unet):
    """
    This is the LIGHT U-Net architecture with residual connections.
    """

    def __init__(self, parameters: dict):
        super(light_resunet, self).__init__(parameters, residualConnections=True)
