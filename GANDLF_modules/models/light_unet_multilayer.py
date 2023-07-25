# -*- coding: utf-8 -*-
"""
Implementation of Light UNet
"""
from torch.nn import ModuleList

from GANDLF.models.seg_modules.DownsamplingModule import DownsamplingModule
from GANDLF.models.seg_modules.EncodingModule import EncodingModule
from GANDLF.models.seg_modules.DecodingModule import DecodingModule
from GANDLF.models.seg_modules.UpsamplingModule import UpsamplingModule
from GANDLF.models.seg_modules.InitialConv import InitialConv
from GANDLF.models.seg_modules.out_conv import out_conv
from .modelBase import ModelBase


class light_unet_multilayer(ModelBase):
    """
    This is the LIGHT U-Net architecture.
    """

    def __init__(
        self,
        parameters: dict,
        residualConnections=False,
    ):
        self.network_kwargs = {"res": residualConnections}
        super(light_unet_multilayer, self).__init__(parameters)

        parameters["model"]["depth"] = parameters["model"].get("depth", 4)

        self.depth = self.model_depth_check(parameters)

        self.ins = InitialConv(
            input_channels=self.n_channels,
            output_channels=self.base_filters,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )

        self.ds = ModuleList([])
        self.en = ModuleList([])
        self.us = ModuleList([])
        self.de = ModuleList([])

        for _ in range(0, self.depth):
            self.ds.append(
                DownsamplingModule(
                    input_channels=self.base_filters,
                    output_channels=self.base_filters,
                    conv=self.Conv,
                    norm=self.Norm,
                )
            )

            self.us.append(
                UpsamplingModule(
                    input_channels=self.base_filters,
                    output_channels=self.base_filters,
                    conv=self.Conv,
                    interpolation_mode=self.linear_interpolation_mode,
                )
            )

            self.de.append(
                DecodingModule(
                    input_channels=self.base_filters * 2,
                    output_channels=self.base_filters,
                    conv=self.Conv,
                    norm=self.Norm,
                    network_kwargs=self.network_kwargs,
                )
            )

            self.en.append(
                EncodingModule(
                    input_channels=self.base_filters,
                    output_channels=self.base_filters,
                    conv=self.Conv,
                    dropout=self.Dropout,
                    norm=self.Norm,
                    network_kwargs=self.network_kwargs,
                )
            )

        self.out = out_conv(
            input_channels=self.base_filters,
            output_channels=self.n_classes,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
            final_convolution_layer=self.final_convolution_layer,
            sigmoid_input_multiplier=self.sigmoid_input_multiplier,
        )

        if "converter_type" in parameters["model"]:
            self.ins = self.converter(self.ins).model
            self.out = self.converter(self.out).model
            for i_lay in range(0, self.depth):
                self.ds[i_lay] = self.converter(self.ds[i_lay]).model
                self.us[i_lay] = self.converter(self.us[i_lay]).model
                self.de[i_lay] = self.converter(self.de[i_lay]).model
                self.en[i_lay] = self.converter(self.en[i_lay]).model

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
        y = []
        y.append(self.ins(x))

        # [downsample --> encode] x num layers
        for i in range(0, self.depth):
            temp = self.ds[i](y[i])
            y.append(self.en[i](temp))

        x = y[-1]

        # [upsample --> encode] x num layers
        for i in range(self.depth - 1, -1, -1):
            x = self.us[i](x)
            x = self.de[i](x, y[i])

        x = self.out(x)
        return x


class light_resunet_multilayer(light_unet_multilayer):
    """
    This is the LIGHT U-Net architecture with residual connections.
    """

    def __init__(self, parameters: dict):
        super(light_resunet_multilayer, self).__init__(
            parameters, residualConnections=True
        )
