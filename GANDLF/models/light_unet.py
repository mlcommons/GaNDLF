# -*- coding: utf-8 -*-
"""
Implementation of UNet
"""

from GANDLF.models.seg_modules.DownsamplingModule import DownsamplingModule
from GANDLF.models.seg_modules.EncodingModule import EncodingModule
from GANDLF.models.seg_modules.DecodingModule import DecodingModule
from GANDLF.models.seg_modules.UpsamplingModule import UpsamplingModule
from GANDLF.models.seg_modules.InitialConv import InitialConv
from GANDLF.models.seg_modules.out_conv import out_conv
from .modelBase import ModelBase
from GANDLF.utils.generic import checkPatchDivisibility


class light_unet(ModelBase):
    """
    This is the LIGHT U-Net architecture.
    """

    def __init__(self, parameters: dict, residualConnections=False):
        self.network_kwargs = {"res": residualConnections}
        super(light_unet, self).__init__(parameters)

        self.network_kwargs = {"res": False}

        assert checkPatchDivisibility(parameters["patch_size"]) == True, (
            "The patch size is not divisible by 16, which is required for "
            + parameters["model"]["architecture"]
        )

        self.ins = InitialConv(
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
            interpolation_mode=self.linear_interpolation_mode,
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
            interpolation_mode=self.linear_interpolation_mode,
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
            interpolation_mode=self.linear_interpolation_mode,
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
            interpolation_mode=self.linear_interpolation_mode,
        )
        # Important to take a look here, this looks confusing but it is like the original implementation of unet
        self.de_0 = DecodingModule(
            input_channels=self.base_filters * 2,
            output_channels=self.base_filters * 2,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.out = out_conv(
            input_channels=self.base_filters * 2,
            output_channels=self.n_classes,
            conv=self.Conv,
            norm=self.Norm,
            final_convolution_layer=self.final_convolution_layer,
            sigmoid_input_multiplier=self.sigmoid_input_multiplier,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A 5D tensor of shape [batch_size, channels, x_dims, y_dims, z_dims].

        Returns:
            torch.Tensor: A 5D tensor of shape [batch_size, n_classes, x_dims, y_dims, z_dims], representing the output of the model for image segmentation.

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
        x = self.de_0(x, x1)
        x = self.out(x)

        return x


class light_resunet(light_unet):
    """
    This is the LIGHT U-Net architecture with residual connections.
    """

    def __init__(self, parameters: dict):
        super(light_resunet, self).__init__(parameters, residualConnections=True)
