import torch.nn.functional as F
import torch.nn as nn
import torch
from models.seg_modules.ResNetModule import ResNetModule
from models.seg_modules.InceptionModule import InceptionModule
from models.seg_modules.IncDownsamplingModule import IncDownsamplingModule
from models.seg_modules.IncUpsamplingModule import IncUpsamplingModule
from models.seg_modules.IncConv import IncConv
from models.seg_modules.ResNetModule import ResNetModule
from models.seg_modules.IncDropout import IncDropout
   
class uinc(nn.Module):
    """
    This is the implementation of the following paper: https://arxiv.org/abs/1907.02110 (from CBICA). Please look at the seg_module files (towards the end), to get 
    a better sense of the Inception Module implemented
    The res parameter is for the addition of the initial feature map with the final feature map after performance of the convolution. 
    For the decoding module, not the initial input but the input after the first convolution is addded to the final output since the initial input and 
    the final one do not have the same dimensions. 
    """  
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
