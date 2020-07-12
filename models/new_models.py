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


def add_conv_block(in_ch=1, out_ch=1, kernel_size=3, dilate=1, last=False, volumetric=True):
    """
    Helpder function
    """
    if volumetric:
        Conv = nn.Conv3d
        BatchNorm = nn.BatchNorm3d
    else:
        Conv = nn.Conv2d
        BatchNorm = nn.BatchNorm2d
    pad = dilate if not last else 0
    conv_1 = Conv(in_ch, out_ch, kernel_size, padding=pad, dilation=dilate)
    bn_1 = BatchNorm(out_ch)

    return [conv_1, bn_1]


class MSDNet(nn.Module):
    """
    Paper: A mixed-scale dense convolutional neural network for image analysis
    Published: PNAS, Jan. 2018 
    Paper: http://www.pnas.org/content/early/2017/12/21/1715832114
    Derived from Shubham Dokania's https://github.com/shubham1810/MS-D_Net_PyTorch
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal(m, m.weight.data)

    def __init__(self, in_channels=1, out_channels=2, num_layers=40, volumetric=True):

        super().__init__()

        self.layer_list = add_conv_block(in_ch=in_channels, volumetric=volumetric)
        
        current_in_channels = 1
        # Add N layers
        for i in range(num_layers):
            s = i % 10 + 1
            self.layer_list += add_conv_block(
                in_ch=current_in_channels,
                dilate=s,
                volumetric=volumetric
            )
            current_in_channels += 1

        # Add final output block
        self.layer_list += add_conv_block(
            in_ch=current_in_channels + in_channels,
            out_ch=out_channels,
            kernel_size=1,
            last=True,
            volumetric=volumetric
        )

        # Add to Module List
        self.layers = nn.ModuleList(self.layer_list)

        self.apply(self.weight_init)

    def forward(self, x):
        prev_features = []
        inp = x
        
        for i, f in enumerate(self.layers):
            # Check if last conv block
            if i == len(self.layers) - 2:
                x = torch.cat(prev_features + [inp], 1)
            
            x = f(x)

            if (i + 1) % 2 == 0 and not i == (len(self.layers) - 1):
                x = F.relu(x)
                # Append output into previous features
                prev_features.append(x)
                x = torch.cat(prev_features, 1)
        return x