# -*- coding: utf-8 -*-
"""
Implementation of UNet with Inception Convolutions - UInc
"""

from GANDLF.models.seg_modules.DownsamplingModule import DownsamplingModule
from GANDLF.models.seg_modules.EncodingModule import EncodingModule
from GANDLF.models.seg_modules.DecodingModule import DecodingModule
from GANDLF.models.seg_modules.UpsamplingModule import UpsamplingModule
from GANDLF.models.seg_modules.in_conv import in_conv
from GANDLF.models.seg_modules.out_conv import out_conv
from .modelBase import ModelBase

class unet(ModelBase):
    """
    This is the standard U-Net architecture : https://arxiv.org/pdf/1606.06650.pdf - without the residual connections. The Downsampling, Encoding, Decoding modules
    are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input channels (filters) and the output channels (filters),
    and some other hyperparameters, which remain constant all the modules. For more details on the smaller modules please have a look at the seg_modules file.
    """
    def __init__(self, n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer):
        super(unet, self).__init__(n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer)
        self.ins = in_conv(n_channels, base_filters, self.Convolution, self.Dropout, self.InstanceNorm)
        self.ds_0 = DownsamplingModule(base_filters, base_filters*2, self.Convolution, self.Dropout, self.InstanceNorm)
        self.en_1 = EncodingModule(base_filters*2, base_filters*2)
        self.ds_1 = DownsamplingModule(base_filters*2, base_filters*4)
        self.en_2 = EncodingModule(base_filters*4, base_filters*4)
        self.ds_2 = DownsamplingModule(base_filters*4, base_filters*8)
        self.en_3 = EncodingModule(base_filters*8, base_filters*8)
        self.ds_3 = DownsamplingModule(base_filters*8, base_filters*16)
        self.en_4 = EncodingModule(base_filters*16, base_filters*16)
        self.us_3 = UpsamplingModule(base_filters*16, base_filters*8)
        self.de_3 = DecodingModule(base_filters*16, base_filters*8)
        self.us_2 = UpsamplingModule(base_filters*8, base_filters*4)
        self.de_2 = DecodingModule(base_filters*8, base_filters*4)
        self.us_1 = UpsamplingModule(base_filters*4, base_filters*2)
        self.de_1 = DecodingModule(base_filters*4, base_filters*2)
        self.us_0 = UpsamplingModule(base_filters*2, base_filters)
        self.out = out_conv(base_filters*2, n_classes, final_convolution_layer = self.final_convolution_layer)

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
