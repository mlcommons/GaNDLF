# -*- coding: utf-8 -*-
"""
Implementation of UNet with deep supervision and residual connections
"""

from GANDLF.models.seg_modules.DownsamplingModule import DownsamplingModule
from GANDLF.models.seg_modules.EncodingModule import EncodingModule
from GANDLF.models.seg_modules.DecodingModule import DecodingModule
from GANDLF.models.seg_modules.UpsamplingModule import UpsamplingModule
from GANDLF.models.seg_modules.InitialConv import InitialConv
from GANDLF.models.seg_modules.out_conv import out_conv
from .modelBase import ModelBase
from GANDLF.utils.generic import checkPatchDivisibility


class deep_unet(ModelBase):
    """
    This is the standard U-Net architecture : https://arxiv.org/pdf/1606.06650.pdf. The 'residualConnections' flag controls residual connections, the
    Downsampling, Encoding, Decoding modules are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input
    channels (filters) and the output channels (filters), and some other hyperparameters, which remain constant all the modules. For more details on the
    smaller modules please have a look at the seg_modules file.
    """

    def __init__(self, parameters: dict, residualConnections=False):
        self.network_kwargs = {"res": residualConnections}
        super(deep_unet, self).__init__(parameters)

        assert checkPatchDivisibility(parameters["patch_size"]) == True, (
            "The patch size is not divisible by 16, which is required for "
            + parameters["model"]["architecture"]
        )

        # Define layers of the UNet model
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
            output_channels=self.base_filters,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.out_3 = out_conv(
            input_channels=self.base_filters * 8,
            output_channels=self.n_classes,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
            final_convolution_layer=self.final_convolution_layer,
            sigmoid_input_multiplier=self.sigmoid_input_multiplier,
        )
        self.out_2 = out_conv(
            input_channels=self.base_filters * 4,
            output_channels=self.n_classes,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
            final_convolution_layer=self.final_convolution_layer,
            sigmoid_input_multiplier=self.sigmoid_input_multiplier,
        )
        self.out_1 = out_conv(
            input_channels=self.base_filters * 2,
            output_channels=self.n_classes,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
            final_convolution_layer=self.final_convolution_layer,
            sigmoid_input_multiplier=self.sigmoid_input_multiplier,
        )
        self.out_0 = out_conv(
            input_channels=self.base_filters,
            output_channels=self.n_classes,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
            final_convolution_layer=self.final_convolution_layer,
            sigmoid_input_multiplier=self.sigmoid_input_multiplier,
        )

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Args:
            x (Tensor): Input Tensor with shape [batch_size, channels, x_dims, y_dims, z_dims].

        Returns:
            (list): List of output Tensors with shape [batch_size, n_classes, x_dims, y_dims, z_dims].
                    The length of the list corresponds to the number of layers in the decoder path.
        """
        # Encoding path
        x1 = self.ins(x)  # First convolution layer

        # Downsample and apply convolution
        x2 = self.ds_0(x1)
        x2 = self.en_1(x2)

        # Downsample and apply convolution
        x3 = self.ds_1(x2)
        x3 = self.en_2(x3)

        # Downsample and apply convolution
        x4 = self.ds_2(x3)
        x4 = self.en_3(x4)

        # Downsample and apply convolution
        x5 = self.ds_3(x4)
        x5 = self.en_4(x5)

        # Decoding path
        # Upsample, concatenate with x4, and apply convolution
        x = self.us_3(x5)
        xl4 = self.de_3(x, x4)
        out_3 = self.out_3(xl4)  # Output tensor level 3

        # Upsample, concatenate with x3, and apply convolution
        x = self.us_2(xl4)
        xl3 = self.de_2(x, x3)
        out_2 = self.out_2(xl3)  # Output tensor level 2

        # Upsample, concatenate with x2, and apply convolution
        x = self.us_1(xl3)
        xl2 = self.de_1(x, x2)
        out_1 = self.out_1(xl2)  # Output tensor level 1

        # Upsample, concatenate with x2, and apply convolution
        x = self.us_0(xl2)
        xl1 = self.de_0(x, x1)
        out_0 = self.out_0(xl1)  # Output tensor level 0

        # Return the 4 output tensors as a list
        return [out_0, out_1, out_2, out_3]


class deep_resunet(deep_unet):
    """
    This is the standard U-Net architecture with residual connections : https://arxiv.org/pdf/1606.06650.pdf.
    The 'residualConnections' flag controls residual connections The Downsampling, Encoding, Decoding modules are defined in the seg_modules file.
    These smaller modules are basically defined by 2 parameters, the input channels (filters) and the output channels (filters),
    and some other hyperparameters, which remain constant all the modules. For more details on the smaller modules please have a look at the seg_modules file.
    """

    def __init__(self, parameters: dict):
        super(deep_resunet, self).__init__(parameters, residualConnections=True)
