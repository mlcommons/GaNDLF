import torch.nn.functional as F
import torch.nn as nn
import torch
from seg_modules import *

class fcn(nn.Module):
    """
    This is the standard FCN (Fully Convolutional Network) architecture : https://arxiv.org/abs/1411.4038 . The Downsampling, Encoding, Decoding modules
    are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input channels (filters) and the output channels (filters),
    and some other hyperparameters, which remain constant all the modules. For more details on the smaller modules please have a look at the seg_modules file.
    """
    def __init__(self, n_channels, n_classes, base_filters):
        super(fcn, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.ins = in_conv(self.n_channels, base_filters)
        self.ds_0 = DownsamplingModule(base_filters, base_filters*2)
        self.en_1 = EncodingModule(base_filters*2, base_filters*2)
        self.ds_1 = DownsamplingModule(base_filters*2, base_filters*4)
        self.en_2 = EncodingModule(base_filters*4, base_filters*4)
        self.ds_2 = DownsamplingModule(base_filters*4, base_filters*8)
        self.en_3 = EncodingModule(base_filters*8, base_filters*8)
        self.ds_3 = DownsamplingModule(base_filters*8, base_filters*16)
        self.en_4 = EncodingModule(base_filters*16, base_filters*16)
        self.us_4 = FCNUpsamplingModule(base_filters*16, 1, scale_factor=5)
        self.us_3 = FCNUpsamplingModule(base_filters*8, 1, scale_factor=4)
        self.us_2 = FCNUpsamplingModule(base_filters*4, 1, scale_factor=3)
        self.us_1 = FCNUpsamplingModule(base_filters*2, 1, scale_factor=2)
        self.us_0 = FCNUpsamplingModule(base_filters, 1, scale_factor=1)
        self.conv_0 = nn.Conv3d(in_channels=5, out_channels=self.n_classes,
                                kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
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
        return F.softmax(x,dim=1)