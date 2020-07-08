import torch.nn.functional as F
import torch.nn as nn
import torch
from seg_modules import in_conv, DownsamplingModule, EncodingModule, InceptionModule, ResNetModule
from seg_modules import UpsamplingModule, DecodingModule,IncDownsamplingModule,IncConv
from seg_modules import out_conv, FCNUpsamplingModule, IncDropout,IncUpsamplingModule

"""
The smaller individual modules of these networks (the ones defined below), are taken from the seg_modules files as seen in the imports above.

"""
'''
This is the standard U-Net architecture : https://arxiv.org/pdf/1606.06650.pdf - without the residual connections. The Downsampling, Encoding, Decoding modules
are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input channels (filters) and the output channels (filters),
and some other hyperparameters, which remain constant all the modules. For more details on the smaller modules please have a look at the seg_modules file.
'''
class unet(nn.Module):
    def __init__(self, n_channels, n_classes, base_filters):
        super(unet, self).__init__()
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
        self.us_3 = UpsamplingModule(base_filters*16, base_filters*8)
        self.de_3 = DecodingModule(base_filters*16, base_filters*8)
        self.us_2 = UpsamplingModule(base_filters*8, base_filters*4)
        self.de_2 = DecodingModule(base_filters*8, base_filters*4)
        self.us_1 = UpsamplingModule(base_filters*4, base_filters*2)
        self.de_1 = DecodingModule(base_filters*4, base_filters*2)
        self.us_0 = UpsamplingModule(base_filters*2, base_filters)
        self.out = out_conv(base_filters*2, self.n_classes)

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

        x = self.us_3(x5)
        x = self.de_3(x, x4)
        x = self.us_2(x)
        x = self.de_2(x, x3)
        x = self.us_1(x)
        x = self.de_1(x, x2)
        x = self.us_0(x)
        x = self.out(x, x1)
        return x

'''
This is the standard U-Net architecture : https://arxiv.org/pdf/1606.06650.pdf. The Downsampling, Encoding, Decoding modules
are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input channels (filters) and the output channels (filters),
and some other hyperparameters, which remain constant all the modules. For more details on the smaller modules please have a look at the seg_modules file.
'''
class resunet(nn.Module):
    def __init__(self, n_channels, n_classes, base_filters):
        super(resunet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.ins = in_conv(self.n_channels, base_filters, res=True)
        self.ds_0 = DownsamplingModule(base_filters, base_filters*2)
        self.en_1 = EncodingModule(base_filters*2, base_filters*2, res=True)
        self.ds_1 = DownsamplingModule(base_filters*2, base_filters*4)
        self.en_2 = EncodingModule(base_filters*4, base_filters*4, res=True)
        self.ds_2 = DownsamplingModule(base_filters*4, base_filters*8)
        self.en_3 = EncodingModule(base_filters*8, base_filters*8, res=True)
        self.ds_3 = DownsamplingModule(base_filters*8, base_filters*16)
        self.en_4 = EncodingModule(base_filters*16, base_filters*16, res=True)
        self.us_3 = UpsamplingModule(base_filters*16, base_filters*8)
        self.de_3 = DecodingModule(base_filters*16, base_filters*8, res=True)
        self.us_2 = UpsamplingModule(base_filters*8, base_filters*4)
        self.de_2 = DecodingModule(base_filters*8, base_filters*4, res=True)
        self.us_1 = UpsamplingModule(base_filters*4, base_filters*2)
        self.de_1 = DecodingModule(base_filters*4, base_filters*2, res=True)
        self.us_0 = UpsamplingModule(base_filters*2, base_filters)
        self.out = out_conv(base_filters*2, self.n_classes, res=True)

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

        x = self.us_3(x5)
        x = self.de_3(x, x4)
        x = self.us_2(x)
        x = self.de_2(x, x3)
        x = self.us_1(x)
        x = self.de_1(x, x2)
        x = self.us_0(x)
        x = self.out(x, x1)
        return x
'''
This is the standard FCN (Fully Convolutional Network) architecture : https://arxiv.org/abs/1411.4038 . The Downsampling, Encoding, Decoding modules
are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input channels (filters) and the output channels (filters),
and some other hyperparameters, which remain constant all the modules. For more details on the smaller modules please have a look at the seg_modules file.
'''

class fcn(nn.Module):
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

'''
This is the implementation of the following paper: https://arxiv.org/abs/1907.02110 (from CBICA). Please look at the seg_module files (towards the end), to get 
a better sense of the Inception Module implemented
The res parameter is for the addition of the initial feature map with the final feature map after performance of the convolution. 
For the decoding module, not the initial input but the input after the first convolution is addded to the final output since the initial input and 
the final one do not have the same dimensions. 
'''     
class uinc(nn.Module):
    def __init__(self,n_channels,n_classes,base_filters):
        super(uinc,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.conv0_1x1 = IncConv(n_channels, base_filters)
        self.rn_0 = ResNetModule(base_filters,base_filters,res=True)
        self.ri_0 = InceptionModule(base_filters,base_filters,res=True)
        self.ds_0 = IncDownsamplingModule(base_filters,base_filters*2)
        self.ri_1 = InceptionModule(base_filters*2,base_filters*2,res=True)
        self.ds_1 = IncDownsamplingModule(base_filters*2,base_filters*4)
        self.ri_2 = InceptionModule(base_filters*4,base_filters*4,res=True)
        self.ds_2 = IncDownsamplingModule(base_filters*4,base_filters*8)
        self.ri_3 = InceptionModule(base_filters*8,base_filters*8,res=True)
        self.ds_3 = IncDownsamplingModule(base_filters*8,base_filters*16)
        self.ri_4 = InceptionModule(base_filters*16,base_filters*16,res=True)
        self.us_3 = IncUpsamplingModule(base_filters*16,base_filters*8)
        self.ri_5 = InceptionModule(base_filters*16,base_filters*16,res=True)
        self.us_2 = IncUpsamplingModule(base_filters*16,base_filters*4)
        self.ri_6 = InceptionModule(base_filters*8,base_filters*8,res=True)
        self.us_1 = IncUpsamplingModule(base_filters*8,base_filters*2)
        self.ri_7 = InceptionModule(base_filters*4,base_filters*4,res=True)
        self.us_0 = IncUpsamplingModule(base_filters*4,base_filters)
        self.ri_8 = InceptionModule(base_filters*2,base_filters*2,res=True)
        self.conv9_1x1 = IncConv(base_filters*2,base_filters)
        self.rn_10 = ResNetModule(base_filters*2,base_filters*2,res=True)
        self.dropout = IncDropout(base_filters*2,n_classes)
    
    def forward(self,x):

        x = self.conv0_1x1(x)
        x1 = self.rn_0(x)
        x2 = self.ri_0(x1)
        x3 = self.ds_0(x2)
        x3 = self.ri_1(x3)
        x4 = self.ds_1(x3)
        x4 = self.ri_2(x4)
        x5 = self.ds_2(x4)
        x5 = self.ri_3(x5)
        x6 = self.ds_3(x5)
        x6 = self.ri_4(x6)
        x6 = self.us_3(x6)
        x6 = self.ri_5(torch.cat((x5,x6),dim=1))
        x6 = self.us_2(x6)
        x6 = self.ri_6(torch.cat((x4,x6),dim=1))
        x6 = self.us_1(x6)
        x6 = self.ri_7(torch.cat((x3,x6),dim=1))
        x6 = self.us_0(x6)
        x6 = self.ri_8(torch.cat((x2,x6),dim=1))
        x6 = self.conv9_1x1(x6)
        x6 = self.rn_10(torch.cat((x1,x6),dim=1))
        x6 = self.dropout(x6)
        x6 = F.softmax(x6,dim=1)
        return x6
