import torch
import torch.nn as nn
import torch.nn.functional as F

class IncConv(nn.Module):
    def __init__(self,input_channels,output_channels,dropout_p=0.3,leakiness=1e-2,conv_bias=True,inst_norm_affine=True,res=False,lrelu_inplace=True):
        nn.Module.__init__(self)
        self.output_channels = output_channels 
        self.leakiness = leakiness
        self.conv_bias = conv_bias 
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.inst_norm = nn.InstanceNorm3d(output_channels,affine=self.inst_norm_affine,track_running_stats = True)
        self.conv = nn.Conv3d(input_channels,output_channels,kernel_size=1,stride=1,padding=0,bias=self.conv_bias)
    def forward(self,x):
        x = F.leaky_relu(self.inst_norm(self.conv(x)),negative_slope = self.leakiness,inplace = self.lrelu_inplace)
        return x
