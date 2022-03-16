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


def checkPatchDimensions(patch_size, numlay):
    if isinstance(patch_size, int):
        patch_size_to_check = np.array(patch_size)
    else:
        patch_size_to_check = patch_size
    # for 2D, don't check divisibility of last dimension
    if patch_size_to_check[-1] == 1:
        patch_size_to_check = patch_size_to_check[:-1]

    if all([x >= 2**numlay and x % 2**numlay == 0 for x in patch_size_to_check]):
        return numlay
    else:
        base2 = np.array(
            [getBase2(x) for x in patch_size_to_check]
        )  # get largest possible number of layers for each dim
        return int(np.min(base2))


def getBase2(num):
    base = 0
    while num % 2 == 0:
        num = num / 2
        base = base + 1
    return base


class unet(ModelBase):
    """
    This is the standard U-Net architecture : https://arxiv.org/pdf/1606.06650.pdf. The 'residualConnections' flag controls residual connections The Downsampling, Encoding, Decoding modules
    are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input channels (filters) and the output channels (filters),
    and some other hyperparameters, which remain constant all the modules. For more details on the smaller modules please have a look at the seg_modules file.
    """

    def __init__(
        self,
        parameters: dict,
        residualConnections=False,
    ):
        self.network_kwargs = {"res": residualConnections}
        super(unet, self).__init__(parameters)

        if not ("depth" in parameters["model"]):
            parameters["model"]["depth"] = 4
            print("Default depth set to 4.")

        patch_check = checkPatchDimensions(
            parameters["patch_size"], numlay=parameters["model"]["depth"]
        )

        if patch_check != parameters["model"]["depth"] and patch_check >= 1:
            print(
                "The patch size is not large enough for desired number of layers. It is expected that each dimension of the patch size is divisible by 2^i, where i is in a integer greater than or equal to 2 Only the first %d layers will run."
                % patch_check
            )

        elif patch_check != parameters["model"]["depth"] and patch_check <= 1:
            sys.exit(
                "The patch size is not large enough for desired number of layers. It is expected that each dimension of the patch size is divisible by 2^i, where i is in a integer greater than or equal to 2."
            )

        self.num_layers = patch_check

        self.ins = in_conv(
            input_channels=self.n_channels,
            output_channels=self.base_filters,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )

        self.ds = []
        self.en = []
        self.us = []
        self.de = []

        for i_lay in range(0, self.num_layers):
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

        self.out = out_conv(
            input_channels=self.base_filters,
            output_channels=self.n_classes,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
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
        y = []
        y.append(self.ins(x))

        # [downsample --> encode] x num layers
        for i in range(0, self.num_layers):
            temp = self.ds[i](y[i])
            y.append(self.en[i](temp))

        x = y[-1]

        # [upsample --> encode] x num layers
        for i in range(self.num_layers - 1, -1, -1):
            x = self.us[i](x)
            x = self.de[i](x, y[i])

        x = self.out(x)
        return x


class resunet(unet):
    """
    This is the standard U-Net architecture with residual connections : https://arxiv.org/pdf/1606.06650.pdf. The 'residualConnections' flag controls residual connections The Downsampling, Encoding, Decoding modules
    are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input channels (filters) and the output channels (filters),
    and some other hyperparameters, which remain constant all the modules. For more details on the smaller modules please have a look at the seg_modules file.
    """

    def __init__(self, parameters: dict):
        super(resunet, self).__init__(parameters, residualConnections=True)
