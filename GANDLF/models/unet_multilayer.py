# -*- coding: utf-8 -*-
"""
Implementation of UNet
"""
import torch
from torch.nn import ModuleList

from GANDLF.models.seg_modules.DownsamplingModule import DownsamplingModule
from GANDLF.models.seg_modules.EncodingModule import EncodingModule
from GANDLF.models.seg_modules.DecodingModule import DecodingModule
from GANDLF.models.seg_modules.UpsamplingModule import UpsamplingModule
from GANDLF.models.seg_modules.InitialConv import InitialConv
from GANDLF.models.seg_modules.out_conv import out_conv
from .modelBase import ModelBase


class unet_multilayer(ModelBase):
    """
    This is the standard U-Net architecture : https://arxiv.org/pdf/1606.06650.pdf. The 'residualConnections' flag controls residual connections, the
    Downsampling, Encoding, Decoding modules are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input
    channels (filters) and the output channels (filters), and some other hyperparameters, which remain constant all the modules. For more details on the
    smaller modules please have a look at the seg_modules file.
    """

    def __init__(self, parameters: dict, residualConnections: bool = False):
        """
        The constructor for the unet_multilayer class.

        Args:
            parameters (dict): A dictionary containing the model parameters.
            residualConnections (bool, optional): Flag to control residual connections. Defaults to False.
        """
        self.network_kwargs = {"res": residualConnections}
        super(unet_multilayer, self).__init__(parameters)

        # Set the depth of the model based on the input parameters
        # If the depth is not set, then set it to 4
        parameters["model"]["depth"] = parameters["model"].get("depth", 4)
        self.depth = self.model_depth_check(parameters)

        # Create the initial convolution layer
        self.ins = InitialConv(
            input_channels=self.n_channels,
            output_channels=self.base_filters,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )

        # Create lists of downsampling, encoding, decoding and upsampling modules
        self.ds = ModuleList([])
        self.en = ModuleList([])
        self.us = ModuleList([])
        self.de = ModuleList([])

        # Create the required number of downsampling, encoding, decoding and upsampling modules
        for i_lay in range(0, self.depth):
            self.ds.append(
                DownsamplingModule(
                    input_channels=self.base_filters * 2 ** (i_lay),
                    output_channels=self.base_filters * 2 ** (i_lay + 1),
                    conv=self.Conv,
                    norm=self.Norm,
                )
            )

            self.us.append(
                UpsamplingModule(
                    input_channels=self.base_filters * 2 ** (i_lay + 1),
                    output_channels=self.base_filters * 2 ** (i_lay),
                    conv=self.Conv,
                    interpolation_mode=self.linear_interpolation_mode,
                )
            )

            self.de.append(
                DecodingModule(
                    input_channels=self.base_filters * 2 ** (i_lay + 1),
                    output_channels=self.base_filters * 2 ** (i_lay),
                    conv=self.Conv,
                    norm=self.Norm,
                    network_kwargs=self.network_kwargs,
                )
            )

            self.en.append(
                EncodingModule(
                    input_channels=self.base_filters * 2 ** (i_lay + 1),
                    output_channels=self.base_filters * 2 ** (i_lay + 1),
                    conv=self.Conv,
                    dropout=self.Dropout,
                    norm=self.Norm,
                    network_kwargs=self.network_kwargs,
                )
            )

        # Create the final convolution layer
        self.out = out_conv(
            input_channels=self.base_filters,
            output_channels=self.n_classes,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
            final_convolution_layer=self.final_convolution_layer,
            sigmoid_input_multiplier=self.sigmoid_input_multiplier,
        )

        # Check if converter_type is passed in model, generally referring to ACS
        if "converter_type" in parameters["model"]:
            self.ins = self.converter(self.ins).model
            self.out = self.converter(self.out).model
            for i_lay in range(0, self.depth):
                self.ds[i_lay] = self.converter(self.ds[i_lay]).model
                self.us[i_lay] = self.converter(self.us[i_lay]).model
                self.de[i_lay] = self.converter(self.de[i_lay]).model
                self.en[i_lay] = self.converter(self.en[i_lay]).model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Store intermediate feature maps

        y = []
        y.append(self.ins(x))

        # [downsample --> encode] x num layers
        for i in range(0, self.depth):
            temp = self.ds[i](y[i])
            y.append(self.en[i](temp))

        # Get the feature map at the last (deepest) layer
        x = y[-1]

        # [upsample --> encode] x num layers
        for i in range(self.depth - 1, -1, -1):
            # Upsample the feature map to match the size of the corresponding feature map in the encoder
            x = self.us[i](x)

            # Concatenate with the corresponding feature map from the encoder
            x = self.de[i](x, y[i])

        # Get the final output
        x = self.out(x)

        return x


class resunet_multilayer(unet_multilayer):
    """
    This is the standard U-Net architecture with residual connections : https://arxiv.org/pdf/1606.06650.pdf.
    The 'residualConnections' flag controls residual connections, the
    Downsampling, Encoding, Decoding modules are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input
    channels (filters) and the output channels (filters), and some other hyperparameters, which remain constant all the modules. For more details on the
    smaller modules please have a look at the seg_modules file.
    """

    def __init__(self, parameters: dict):
        super(resunet_multilayer, self).__init__(parameters, residualConnections=True)
