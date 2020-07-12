import torch
import torch.nn as nn
import torch.nn.functional as F




''' 
Link to the paper (CBICA): https://arxiv.org/pdf/1907.02110.pdf. Below implemented are the smaller modules that are integrated to form the larger Inc U Net arch.
Architecture is defined on Page 5 Figure 1 of the paper. 
'''
        
'''
This is the module implementation on Page 6 Figure 2 (diagram on the right) of the above mentioned paper. In summary, this consists of 4 parallel pathways each with f/4 feature maps (f is the
number of feature maps of the input to the InceptionModule. These 4 feature maps (or channels) are concatenated after being processed by the Inception Module)
'''
class InceptionModule(nn.Module):
    def __init__(self,input_channels,output_channels,dropout_p=0.3,leakiness=1e-2,conv_bias=True,inst_norm_affine=True,res=False,lrelu_inplace=True):
        nn.Module.__init__(self)
        self.res = res
        self.output_channels = output_channels
        self.dropout_p = dropout_p
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.inst_norm_affine = inst_norm_affine
        self.lrelu_inplace = lrelu_inplace
        self.dropout = nn.Dropout3d(dropout_p)
        self.inst_norm = nn.InstanceNorm3d(int(output_channels/4),affine = self.inst_norm_affine, track_running_stats = True)
        self.inst_norm_final = nn.InstanceNorm3d(output_channels,affine = self.inst_norm_affine, track_running_stats = True)
        self.conv_1x1 = nn.Conv3d(output_channels,int(output_channels/4),kernel_size = 1,stride=1,padding=0,bias = self.conv_bias)
        self.conv_3x3 = nn.Conv3d(int(output_channels/4),int(output_channels/4),kernel_size=3,stride=1,padding=1,bias=self.conv_bias)
        self.conv_1x1_final = nn.Conv3d(output_channels,output_channels,kernel_size = 1, stride = 1, padding=0,bias = self.conv_bias)
        
    def forward(self,x):
        output_channels = self.output_channels
        if self.res == True:
            skip = x
        x1 = F.leaky_relu(self.inst_norm(self.conv_1x1(x)),negative_slope = self.leakiness,inplace = self.lrelu_inplace)
        
        x2 = F.leaky_relu(self.inst_norm(self.conv_1x1(x)),negative_slope = self.leakiness,inplace = self.lrelu_inplace)
        x2 = F.leaky_relu(self.inst_norm(self.conv_3x3(x2)),negative_slope = self.leakiness,inplace = self.lrelu_inplace)
        
        
        x3 = F.leaky_relu(self.inst_norm(self.conv_1x1(x)),negative_slope = self.leakiness,inplace = self.lrelu_inplace)
        x3 = F.leaky_relu(self.inst_norm(self.conv_3x3(x3)),negative_slope = self.leakiness,inplace = self.lrelu_inplace)
        x3 = F.leaky_relu(self.inst_norm(self.conv_3x3(x3)),negative_slope = self.leakiness,inplace = self.lrelu_inplace)
        
        x4 = F.leaky_relu(self.inst_norm(self.conv_1x1(x)),negative_slope = self.leakiness,inplace = self.lrelu_inplace)
        x4 = F.leaky_relu(self.inst_norm(self.conv_3x3(x4)),negative_slope = self.leakiness,inplace = self.lrelu_inplace)
        x4 = F.leaky_relu(self.inst_norm(self.conv_3x3(x4)),negative_slope = self.leakiness,inplace = self.lrelu_inplace)
        x4 = F.leaky_relu(self.inst_norm(self.conv_3x3(x4)),negative_slope = self.leakiness,inplace = self.lrelu_inplace)
        
        x = torch.cat((x1,x2,x3,x4),dim=1)
        x = self.inst_norm_final(self.conv_1x1_final(x))
        
        x = x + skip
        x = F.leaky_relu(x,negative_slope = self.leakiness,inplace = self.lrelu_inplace)
        
        return x 

   '''
   This is the implementation of the Page 6 Figure 2 (diagram on the left)
   '''
    
class ResNetModule(nn.Module):
    def __init__(self,input_channels,output_channels,dropout_p=0.3,leakiness=1e-2,conv_bias=True,inst_norm_affine=True,res=False,lrelu_inplace=True):
        nn.Module.__init__(self)
        self.output_channels = output_channels
        self.dropout_p = dropout_p
        self.leakiness = leakiness
        self.conv_bias = conv_bias 
        self.inst_norm_affine = inst_norm_affine
        self.res = res
        self.lrelu_inplace = lrelu_inplace 
        self.dropout = nn.Dropout3d(dropout_p)
        self.inst_norm = nn.InstanceNorm3d(output_channels,affine = self.inst_norm_affine, track_running_stats = True)
        self.conv = nn.Conv3d(output_channels,output_channels,kernel_size=3,stride=1,padding=1,bias = self.conv_bias)
    def forward(self,x):
        if self.res == True:
            skip = x          
        x = F.leaky_relu(self.inst_norm(self.conv(x)),negative_slope = self.leakiness,inplace = self.lrelu_inplace)
        x = self.inst_norm(self.conv(x))
        x = x + skip
        x = F.leaky_relu(x,negative_slope = self.leakiness, inplace = self.lrelu_inplace)
        return x 

  ''''
  The Upsampling and Downsampling modules given below are same as the ones used above. Just used a different name for clarity, since the overall architecture of 
  the Inception U-Net is significantly different from the other U-net variants. 
  '''
    
class IncDownsamplingModule(nn.Module):
    def __init__(self, input_channels, output_channels, leakiness=1e-2, kernel_size=1, conv_bias=True, 
                 inst_norm_affine=True, lrelu_inplace=True):
        nn.Module.__init__(self)
        self.output_channels = output_channels 
        self.input_channels = input_channels
        self.leakiness = leakiness
        self.conv_bias = conv_bias 
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.inst_norm = nn.InstanceNorm3d(output_channels,affine = self.inst_norm_affine, track_running_stats = True)
        self.down = nn.Conv3d(input_channels,output_channels,kernel_size = 1, stride = 2, padding = 0, bias = self.conv_bias)
    
    def forward(self,x):
        x = F.leaky_relu(self.inst_norm(self.down(x)),negative_slope = self.leakiness, inplace = self.lrelu_inplace)
        return x
    
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

class IncDropout(nn.Module):
    def __init__(self,input_channels,output_channels,dropout_p=0.3,leakiness=1e-2,conv_bias=True,inst_norm_affine=True, res = False, lrelu_inplace = True):
        nn.Module.__init__(self)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dropout_p = dropout_p
        self.leakiness = leakiness
        self.conv_bias = conv_bias 
        self.inst_norm_affine = inst_norm_affine
        self.res = res
        self.lrelu_inplace = lrelu_inplace 
        self.dropout = nn.Dropout3d(dropout_p)
        self.conv = nn.Conv3d(input_channels,output_channels,kernel_size=1,stride=1,padding=0,bias=self.conv_bias)
    def forward(self,x):
        x = self.dropout(x)
        x = self.conv(x)
        return x
class IncUpsamplingModule(nn.Module):
    def __init__(self,input_channels,output_channels,dropout_p=0.3,leakiness=1e-2,conv_bias=True,inst_norm_affine=True, res = False, lrelu_inplace = True):
        nn.Module.__init__(self)
        self.input_channels = input_channels 
        self.output_channels = output_channels
        self.dropout_p = dropout_p
        self.leakiness = leakiness
        self.conv_bias = conv_bias 
        self.inst_norm_affine = inst_norm_affine
        self.res = res
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm = nn.InstanceNorm3d(output_channels,affine = self.inst_norm_affine, track_running_stats = True)
        self.up = nn.ConvTranspose3d(input_channels,output_channels,kernel_size=1,stride=2,padding=0,output_padding =1,bias = self.conv_bias)
    def forward(self,x):
        x = F.leaky_relu(self.inst_norm(self.up(x)),negative_slope=self.leakiness,inplace=self.lrelu_inplace)
        return x
