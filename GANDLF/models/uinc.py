# -*- coding: utf-8 -*-
"""
Implementation of UNet with Inception Convolutions - UInc
"""

import torch.nn.functional as F
import torch
from GANDLF.models.seg_modules.ResNetModule import ResNetModule
from GANDLF.models.seg_modules.InceptionModule import InceptionModule
from GANDLF.models.seg_modules.IncDownsamplingModule import IncDownsamplingModule
from GANDLF.models.seg_modules.IncUpsamplingModule import IncUpsamplingModule
from GANDLF.models.seg_modules.IncConv import IncConv
from GANDLF.models.seg_modules.IncDropout import IncDropout
from .modelBase import ModelBase
from GANDLF.utils.generic import checkPatchDivisibility


class uinc(ModelBase):
    """
    This is the implementation of the following paper: https://arxiv.org/abs/1907.02110.
    Please look at the seg_module files (towards the end), to get  a better sense of
    the Inception Module implemented. The res parameter is for the addition of the initial feature
    map with the final feature map after performance of the convolution. For the decoding module,
    not the initial input but the input after the first convolution is added to the final output
    since the initial input and the final one do not have the same dimensions.
    """

    def __init__(
        self,
        parameters: dict,
    ):
        super(uinc, self).__init__(parameters)

        assert checkPatchDivisibility(parameters["patch_size"]) == True, (
            "The patch size is not divisible by 16, which is required for "
            + parameters["model"]["architecture"]
        )

        assert parameters["model"]["base_filters"] % 4 == 0, (
            "The 'base_filters' should be divisible by '4'"
            + " for "
            + parameters["model"]["architecture"]
        )

        self.conv0_1x1 = IncConv(
            self.n_channels,
            self.base_filters,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
        )
        self.rn_0 = ResNetModule(
            self.base_filters,
            self.base_filters,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=True,
        )
        self.ri_0 = InceptionModule(
            self.base_filters,
            self.base_filters,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=True,
        )
        self.ds_0 = IncDownsamplingModule(
            self.base_filters,
            self.base_filters * 2,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
        )
        self.ri_1 = InceptionModule(
            self.base_filters * 2,
            self.base_filters * 2,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=True,
        )
        self.ds_1 = IncDownsamplingModule(
            self.base_filters * 2,
            self.base_filters * 4,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
        )
        self.ri_2 = InceptionModule(
            self.base_filters * 4,
            self.base_filters * 4,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=True,
        )
        self.ds_2 = IncDownsamplingModule(
            self.base_filters * 4,
            self.base_filters * 8,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
        )
        self.ri_3 = InceptionModule(
            self.base_filters * 8,
            self.base_filters * 8,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=True,
        )
        self.ds_3 = IncDownsamplingModule(
            self.base_filters * 8,
            self.base_filters * 16,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
        )
        self.ri_4 = InceptionModule(
            self.base_filters * 16,
            self.base_filters * 16,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=True,
        )
        self.us_3 = IncUpsamplingModule(
            self.base_filters * 16,
            self.base_filters * 8,
            self.ConvTranspose,
            self.Dropout,
            self.InstanceNorm,
        )
        self.ri_5 = InceptionModule(
            self.base_filters * 16,
            self.base_filters * 16,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=True,
        )
        self.us_2 = IncUpsamplingModule(
            self.base_filters * 16,
            self.base_filters * 4,
            self.ConvTranspose,
            self.Dropout,
            self.InstanceNorm,
        )
        self.ri_6 = InceptionModule(
            self.base_filters * 8,
            self.base_filters * 8,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=True,
        )
        self.us_1 = IncUpsamplingModule(
            self.base_filters * 8,
            self.base_filters * 2,
            self.ConvTranspose,
            self.Dropout,
            self.InstanceNorm,
        )
        self.ri_7 = InceptionModule(
            self.base_filters * 4,
            self.base_filters * 4,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=True,
        )
        self.us_0 = IncUpsamplingModule(
            self.base_filters * 4,
            self.base_filters,
            self.ConvTranspose,
            self.Dropout,
            self.InstanceNorm,
        )
        self.ri_8 = InceptionModule(
            self.base_filters * 2,
            self.base_filters * 2,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=True,
        )
        self.conv9_1x1 = IncConv(
            self.base_filters * 2,
            self.base_filters,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
        )
        self.rn_10 = ResNetModule(
            self.base_filters * 2,
            self.base_filters * 2,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
            res=True,
        )
        self.dropout = IncDropout(
            self.base_filters * 2,
            self.n_classes,
            self.Conv,
            self.Dropout,
            self.InstanceNorm,
        )

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): Should be a 5D Tensor as [batch_size, channels, x_dims, y_dims, z_dims].

        Returns:
            x6 (torch.Tensor): Returns a 5D Output Tensor as [batch_size, n_classes, x_dims, y_dims, z_dims].

        """
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
        x6 = self.ri_5(torch.cat((x5, x6), dim=1))
        x6 = self.us_2(x6)
        x6 = self.ri_6(torch.cat((x4, x6), dim=1))
        x6 = self.us_1(x6)
        x6 = self.ri_7(torch.cat((x3, x6), dim=1))
        x6 = self.us_0(x6)
        x6 = self.ri_8(torch.cat((x2, x6), dim=1))
        x6 = self.conv9_1x1(x6)
        x6 = self.rn_10(torch.cat((x1, x6), dim=1))
        x6 = self.dropout(x6)

        if not self.final_convolution_layer is None:
            if self.final_convolution_layer == F.softmax:
                x6 = self.final_convolution_layer(x6, dim=1)
            else:
                x6 = self.final_convolution_layer(x6)

        return x6
