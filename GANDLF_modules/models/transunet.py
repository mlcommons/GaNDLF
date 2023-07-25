# -*- coding: utf-8 -*-
"""
Implementation of TransUNet
"""

from GANDLF.models.seg_modules.DownsamplingModule import DownsamplingModule
from GANDLF.models.seg_modules.EncodingModule import EncodingModule
from GANDLF.models.seg_modules.InitialConv import InitialConv
from GANDLF.models.seg_modules.out_conv import out_conv
from .modelBase import ModelBase
import torch
import torch.nn as nn
from torch.nn import ModuleList
from .unetr import _Transformer


class _DecoderCUP(nn.Sequential):
    """
    Decoder module used in the U-Net architecture for upsampling the encoded feature maps.

    Args:
        in_feats (int): Number of input channels.
        out_feats (int): Number of output channels.
        Norm (nn.Module): Normalization layer to be applied after convolution.
        Conv (nn.Module): Convolutional layer used in the decoder.
        Upsample (nn.Module): Upsampling layer used to increase the spatial resolution of the feature maps.

    """

    def __init__(self, in_feats, out_feats, Norm, Conv, Upsample):
        super().__init__()

        self.conv = Conv(
            in_channels=in_feats,
            out_channels=out_feats,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.norm = Norm(out_feats)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = Upsample

    def forward(self, x1, x2):
        """
        Forward pass of the decoder module.

        Args:
            x1 (torch.Tensor): Tensor with shape (batch_size, in_feats, H, W),
                               where H and W are the height and width of the input tensor.
            x2 (torch.Tensor): Tensor with shape (batch_size, out_feats, 2*H, 2*W),
                               where H and W are the height and width of the input tensor.

        Returns:
            x (torch.Tensor): Tensor with shape (batch_size, out_feats, 2*H, 2*W),
                              where H and W are the height and width of the input tensor.
        """
        if x2 is not None:
            x1 = torch.cat([x1, x2], dim=1)

        x = self.conv(x1)
        x = self.norm(x)
        x = self.relu(x)

        x = self.upsample(x)

        return x


class transunet(ModelBase):
    """
    transunet architecture: https://doi.org/10.48550/arXiv.2102.04306

    This class implements the TransUNet model for medical image segmentation tasks.
    The TransUNet architecture consists of an encoder-decoder structure, including downsampling,
    encoding, and decoding modules. The 'residualConnections' flag controls residual connections.
    The Downsampling, Encoding, and Decoding modules are defined in the seg_modules file.

    Args:
        parameters (dict): Dictionary containing model parameters and hyperparameters.

    Attributes:
        depth (int): Depth of the model.
        num_heads (int): Number of self-attention heads in the transformer.
        embed_size (int): Embedding dimension for the transformer.
        patch_dim (list): The dimensions of the image patch.
        ins (InitialConv): Initial convolutional layer.
        ds (ModuleList): List containing the downsampling modules.
        en (ModuleList): List containing the encoding modules.
        de (ModuleList): List containing the decoding modules.
        transformer (_Transformer): Transformer module for the architecture.
        transCUP (_DecoderCUP): Decoder CUP module for the architecture.
        out (out_conv): Final output convolutional layer.
    """

    def __init__(
        self,
        parameters: dict,
    ):
        super(transunet, self).__init__(parameters)

        # Initialize default parameters if not found
        parameters["model"]["depth"] = parameters["model"].get("depth", 4)
        parameters["model"]["num_heads"] = parameters["model"].get("num_heads", 12)
        parameters["model"]["embed_dim"] = parameters["model"].get("embed_dim", 768)

        self.depth = self.model_depth_check(parameters)

        # Set the image size and upsampling method based on the number of dimensions
        if self.n_dimensions == 2:
            self.img_size = parameters["patch_size"][0:2]
            self.upsample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
        elif self.n_dimensions == 3:
            self.img_size = parameters["patch_size"]
            self.upsample = nn.Upsample(
                scale_factor=2, mode="trilinear", align_corners=True
            )

        self.num_layers = 3 * self.depth  # number of transformer layers
        self.out_layers = [self.num_layers - 1]

        self.num_heads = parameters["model"]["num_heads"]
        self.embed_size = parameters["model"]["embed_dim"]

        assert (
            self.embed_size % self.num_heads == 0
        ), "The embedding dimension must be divisible by the number of self-attention heads"

        self.patch_dim = [i // 2 ** (self.depth) for i in self.img_size]

        # Initialize model modules
        self.ins = InitialConv(
            input_channels=self.n_channels,
            output_channels=self.base_filters,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
        )

        # Initialize downsampling, encoding, and decoding modules
        self.ds = ModuleList([])
        self.en = ModuleList([])
        self.de = ModuleList([])

        for i_lay in range(0, self.depth):
            self.ds.append(
                DownsamplingModule(
                    input_channels=self.base_filters * 2 ** (i_lay),
                    output_channels=self.base_filters * 2 ** (i_lay + 1),
                    conv=self.Conv,
                    norm=self.Norm,
                )
            )

            self.de.append(
                _DecoderCUP(
                    in_feats=2 * self.base_filters * 2 ** (i_lay + 1),
                    out_feats=self.base_filters * 2 ** (i_lay),
                    Conv=self.Conv,
                    Norm=self.Norm,
                    Upsample=self.upsample,
                )
            )

            self.en.append(
                EncodingModule(
                    input_channels=self.base_filters * 2 ** (i_lay + 1),
                    output_channels=self.base_filters * 2 ** (i_lay + 1),
                    conv=self.Conv,
                    dropout=self.Dropout,
                    norm=self.Norm,
                )
            )

        # Initialize transformer and decoder CUP modules
        self.transformer = _Transformer(
            img_size=[i // 2 ** (self.depth) for i in self.img_size],
            patch_size=1,
            in_feats=self.base_filters * 2**self.depth,
            embed_size=self.embed_size,
            num_heads=self.num_heads,
            mlp_dim=2048,
            num_layers=self.num_layers,
            out_layers=self.out_layers,
            Conv=self.Conv,
            Norm=self.Norm,
        )

        self.transCUP = _DecoderCUP(
            in_feats=self.embed_size,
            out_feats=self.base_filters * 2 ** (self.depth - 1),
            Conv=self.Conv,
            Norm=self.Norm,
            Upsample=self.upsample,
        )

        # TODO: conv 3x3 --> ReLU --> outconv
        # Initialize final output convolutional layer
        self.out = out_conv(
            input_channels=self.base_filters,
            output_channels=self.n_classes,
            conv=self.Conv,
            norm=self.Norm,
            final_convolution_layer=self.final_convolution_layer,
            sigmoid_input_multiplier=self.sigmoid_input_multiplier,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Should be a 5D Tensor as [batch_size, channels, x_dims, y_dims, z_dims].

        Returns
            x (Tensor): Returns a 5D Output Tensor as [batch_size, n_classes, x_dims, y_dims, z_dims].

        """
        y = []
        y.append(self.ins(x))

        # [downsample --> encode] x num layers
        for i in range(0, self.depth):
            temp = self.ds[i](y[i])
            y.append(self.en[i](temp))

        # Apply transformer and decoder CUP
        x = self.transformer(y[-1])[-1]
        x = x.transpose(-1, -2).view(-1, self.embed_size, *self.patch_dim)
        x = self.transCUP(x, None)

        # [upsample --> encode] x num layers
        for i in range(self.depth - 1, 0, -1):
            x = self.de[i - 1](x, y[i])

        # Final output convolutional layer
        x = self.out(x)
        return x
