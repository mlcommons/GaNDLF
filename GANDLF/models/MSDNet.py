# -*- coding: utf-8 -*-
"""
Implementation of MSDNet

This module contains the implementation of MSDNet, a mixed-scale dense convolutional neural network
for image analysis, as described in the paper "A mixed-scale dense convolutional neural network for image analysis" by Daniel et al.

References
    Daniël M. Pelt and James A. Sethian, A mixed-scale dense convolutional neural network for image analysis.
    Proceedings of the National Academy of Sciences, 115(2), 254-259. doi: 10.1073/pnas.1715832114
    Paper: http://www.pnas.org/content/early/2017/12/21/1715832114
    Derived from Shubham Dokania's implementation: https://github.com/shubham1810/MS-D_Net_PyTorch
"""

import torch.nn.functional as F
import torch.nn as nn
import torch
from GANDLF.models.seg_modules.add_conv_block import add_conv_block
from .modelBase import ModelBase


class MSDNet(ModelBase):
    """
    A pytorch implementation of the Multi-Scale Dense Network.
    """

    @staticmethod
    def weight_init(layer):
        """
        Initializes the weights of a nn.Linear layer using Kaiming Normal initialization.

        Args:
            layer (nn.Linear): A linear layer to initialize.
        """
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer, layer.weight.data)

    def __init__(self, parameters: dict, num_layers=4):
        """
        A multi-scale dense neural network architecture that consists of multiple convolutional layers with different dilation rates.

        Args:
            parameters (dict): A dictionary containing the parameters required to create the convolutional and
        batch normalization layers in the model.
            num_layers (int): The number of layers to be added to the model.

        Returns:
        - None
        """
        super(MSDNet, self).__init__(parameters)

        self.layer_list = add_conv_block(
            self.Conv, self.BatchNorm, in_channels=self.n_channels
        )

        current_in_channels = 1
        # Add N layers
        for i in range(num_layers):
            s = i % 10 + 1
            self.layer_list += add_conv_block(
                self.Conv, self.BatchNorm, in_channels=current_in_channels, dilation=s
            )
            current_in_channels += 1

        # Add final output block
        self.layer_list += add_conv_block(
            self.Conv,
            self.BatchNorm,
            in_channels=current_in_channels + self.n_channels,
            out_channels=self.n_classes,
            kernel_size=1,
            last=True,
        )

        # Add to Module List
        self.layers = nn.ModuleList(self.layer_list)

        self.apply(self.weight_init)

    def forward(self, x):
        """
        Forward pass method for the MSDNet class.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            torch.Tensor: The output tensor from the model.
        """
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
