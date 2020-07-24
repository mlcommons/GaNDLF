import torch.nn.functional as F
import torch.nn as nn
import torch
from DeepSAGE.models.seg_modules import *


class MSDNet(nn.Module):
    """
    Paper: A mixed-scale dense convolutional neural network for image analysis
    Published: PNAS, Jan. 2018 
    Paper: http://www.pnas.org/content/early/2017/12/21/1715832114
    DOI: 10.1073/pnas.1715832114
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