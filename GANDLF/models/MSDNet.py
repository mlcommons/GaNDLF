# -*- coding: utf-8 -*-
"""
Implementation of MSDNet
"""

import torch.nn.functional as F
import torch.nn as nn
import torch
from GANDLF.models.seg_modules.add_conv_block import add_conv_block
from .modelBase import ModelBase

class MSDNet(ModelBase):
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
            nn.init.kaiming_normal_(m, m.weight.data)

    def __init__(self, n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer, num_layers = 4):

        super(MSDNet, self).__init__(n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer)

        self.layer_list = add_conv_block(self.Conv, self.BatchNorm, in_ch=n_channels)

        current_in_channels = 1
        # Add N layers
        for i in range(num_layers):
            s = i % 10 + 1
            self.layer_list += add_conv_block(
                self.Conv,
                self.BatchNorm,
                in_ch=current_in_channels,
                dilate=s
            )
            current_in_channels += 1

        # Add final output block
        self.layer_list += add_conv_block(
            self.Conv,
            self.BatchNorm,
            in_ch=current_in_channels + n_channels,
            out_ch=n_classes,
            kernel_size=1,
            last=True
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
