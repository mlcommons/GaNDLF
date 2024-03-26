# -*- coding: utf-8 -*-
"""
Implementation of UNet
"""
from .modelBase import ModelBase
from GANDLF.models.seg_modules.DownsamplingModule import DownsamplingModule
from GANDLF.models.seg_modules.EncodingModule import EncodingModule
from GANDLF.models.seg_modules.DecodingModule import DecodingModule
from GANDLF.models.seg_modules.UpsamplingModule import UpsamplingModule
from GANDLF.models.seg_modules.InitialConv import InitialConv
from GANDLF.models.seg_modules.out_conv import out_conv
from GANDLF.utils.generic import checkPatchDivisibility


class unet(ModelBase):
    """
    This is the standard U-Net architecture : https://arxiv.org/pdf/1606.06650.pdf. The 'residualConnections' flag controls residual connections, the
    Downsampling, Encoding, Decoding modules are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input
    channels (filters) and the output channels (filters), and some other hyperparameters, which remain constant all the modules. For more details on the
    smaller modules please have a look at the seg_modules file.
    """

    def __init__(self, parameters: dict, residualConnections=False):
        self.network_kwargs = {"res": residualConnections}
        super(unet, self).__init__(parameters)

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
            network_kwargs=self.network_kwargs,
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
            network_kwargs=self.network_kwargs,
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
            network_kwargs=self.network_kwargs,
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
            network_kwargs=self.network_kwargs,
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
            network_kwargs=self.network_kwargs,
        )
        self.us_3 = UpsamplingModule(
            input_channels=self.base_filters * 16,
            output_channels=self.base_filters * 8,
            conv=self.Conv,
            interpolation_mode=self.linear_interpolation_mode,
        )
        self.de_3 = DecodingModule(
            input_channels=self.base_filters * 16,
            output_channels=self.base_filters * 8,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.us_2 = UpsamplingModule(
            input_channels=self.base_filters * 8,
            output_channels=self.base_filters * 4,
            conv=self.Conv,
            interpolation_mode=self.linear_interpolation_mode,
        )
        self.de_2 = DecodingModule(
            input_channels=self.base_filters * 8,
            output_channels=self.base_filters * 4,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.us_1 = UpsamplingModule(
            input_channels=self.base_filters * 4,
            output_channels=self.base_filters * 2,
            conv=self.Conv,
            interpolation_mode=self.linear_interpolation_mode,
        )
        self.de_1 = DecodingModule(
            input_channels=self.base_filters * 4,
            output_channels=self.base_filters * 2,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.us_0 = UpsamplingModule(
            input_channels=self.base_filters * 2,
            output_channels=self.base_filters,
            conv=self.Conv,
            interpolation_mode=self.linear_interpolation_mode,
        )
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
            network_kwargs=self.network_kwargs,
            final_convolution_layer=self.final_convolution_layer,
            sigmoid_input_multiplier=self.sigmoid_input_multiplier,
        )

        if "converter_type" in parameters["model"]:
            self.ins = self.converter(self.ins).model
            self.ds_0 = self.converter(self.ds_0).model
            self.en_1 = self.converter(self.en_1).model
            self.ds_1 = self.converter(self.ds_1).model
            self.en_2 = self.converter(self.en_2).model
            self.ds_2 = self.converter(self.ds_2).model
            self.en_3 = self.converter(self.en_3).model
            self.ds_3 = self.converter(self.ds_3).model
            self.en_4 = self.converter(self.en_4).model
            self.us_3 = self.converter(self.us_3).model
            self.de_3 = self.converter(self.de_3).model
            self.us_2 = self.converter(self.us_2).model
            self.de_2 = self.converter(self.de_2).model
            self.us_1 = self.converter(self.us_1).model
            self.de_1 = self.converter(self.de_1).model
            self.us_0 = self.converter(self.us_0).model
            self.de_0 = self.converter(self.de_0).model
            self.out = self.converter(self.out).model

    def forward(self, x):
        """
        Forward pass of the UNet model.

        Args:
            x (Tensor): Should be a 5D Tensor as [batch_size, channels, x_dims, y_dims, z_dims].

        Returns:
            x (Tensor): Returns a 5D Output Tensor as [batch_size, n_classes, x_dims, y_dims, z_dims].

        """

        # Encoding path
        x1 = self.ins(x)

        # Apply Downsampling and encoding modules
        x2 = self.ds_0(x1)
        x2 = self.en_1(x2)

        # Apply Downsampling and encoding modules
        x3 = self.ds_1(x2)
        x3 = self.en_2(x3)

        # Apply Downsampling and encoding modules
        x4 = self.ds_2(x3)
        x4 = self.en_3(x4)

        # Apply Downsampling and encoding modules
        x5 = self.ds_3(x4)
        x5 = self.en_4(x5)

        # Decoding path
        x = self.us_3(x5)
        x = self.de_3(x, x4)
        x = self.us_2(x)
        x = self.de_2(x, x3)
        x = self.us_1(x)
        x = self.de_1(x, x2)
        x = self.us_0(x)
        x = self.de_0(x, x1)
        x = self.out(x)

        # Return output tensors
        return x


class resunet(unet):
    """
    This is the standard U-Net architecture with residual connections : https://arxiv.org/pdf/1606.06650.pdf.
    The 'residualConnections' flag controls residual connections The Downsampling, Encoding, Decoding modules are defined in the seg_modules file.
    These smaller modules are basically defined by 2 parameters, the input channels (filters) and the output channels (filters),
    and some other hyperparameters, which remain constant all the modules. For more details on the smaller modules please have a look at the seg_modules file.
    """

    def __init__(self, parameters: dict):
        super(resunet, self).__init__(parameters, residualConnections=True)
