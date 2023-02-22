from .modelBase import ModelBase
from GANDLF.models.seg_modules.out_conv import out_conv
import torch
import torch.nn as nn
from torch.nn import ModuleList
import numpy as np
import math


class _DeconvConvBlock(nn.Sequential):
    def __init__(self, in_feats, out_feats, Norm, Conv, Deconv):
        super().__init__()

        self.add_module(
            "deconv",
            Deconv(
                in_channels=in_feats,
                out_channels=out_feats,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

        self.add_module(
            "conv",
            Conv(
                in_channels=out_feats,
                out_channels=out_feats,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        self.add_module("norm", Norm(out_feats))
        self.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.deconv(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class _ConvBlock(nn.Sequential):
    def __init__(self, in_feats, out_feats, Norm, Conv):
        super().__init__()

        self.add_module(
            "conv",
            Conv(
                in_channels=in_feats,
                out_channels=out_feats,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        self.add_module("norm", Norm(out_feats))
        self.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class _UpsampleBlock(nn.Sequential):
    def __init__(self, in_feats, Norm, Conv, Deconv):
        super().__init__()

        self.add_module(
            "conv1",
            Conv(
                in_channels=in_feats,
                out_channels=in_feats // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        self.add_module("norm1", Norm(in_feats // 2))
        self.add_module("relu1", nn.ReLU(inplace=True))

        self.add_module(
            "conv2",
            Conv(
                in_channels=in_feats // 2,
                out_channels=in_feats // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        self.add_module("norm2", Norm(in_feats // 2))
        self.add_module("relu2", nn.ReLU(inplace=True))

        self.add_module(
            "deconv",
            Deconv(
                in_channels=in_feats // 2,
                out_channels=in_feats // 4,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.deconv(x)

        return x


class _MLP(nn.Sequential):
    def __init__(self, in_feats, out_feats, Norm):
        super().__init__()

        self.add_module("norm", nn.LayerNorm(in_feats))

        self.add_module("linear1", nn.Linear(in_feats, out_feats))
        self.add_module("gelu1", nn.GELU())

        self.add_module("linear2", nn.Linear(out_feats, in_feats))
        self.add_module("gelu2", nn.GELU())

    def forward(self, x):
        x = self.norm(x)
        x = self.linear1(x)
        x = self.gelu1(x)

        x = self.linear2(x)
        x = self.gelu2(x)

        return x


class _MSA(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.query_size = int(embed_size / self.num_heads)
        self.all_heads = self.query_size * self.num_heads  # should equal embed_size

        self.query = nn.Linear(embed_size, self.all_heads)
        self.key = nn.Linear(embed_size, self.all_heads)
        self.value = nn.Linear(embed_size, self.all_heads)

        self.out = nn.Linear(self.all_heads, self.all_heads)

        self.softmax = nn.Softmax(dim=-1)

    def reshape(self, x):
        x_shape = list(x.size()[:-1]) + [self.num_heads, self.query_size]
        x = x.view(*x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        query = self.reshape(self.query(x))
        key = self.reshape(self.key(x))
        value = self.reshape(self.value(x))

        attention_weights = self.softmax(
            torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.query_size)
        )
        self_attention = torch.matmul(attention_weights, value)
        self_attention = self_attention.permute(0, 2, 1, 3).contiguous()
        self_attention = self_attention.view(
            list(self_attention.size()[:-2]) + [self.all_heads]
        )  # reshape

        msa = self.out(self_attention)

        return msa


class _Embedding(nn.Module):
    def __init__(self, img_size, patch_size, in_feats, embed_size, Conv):
        super().__init__()
        # self.n_patches = int(np.prod(img_size) / (patch_size**3))
        self.n_patches = int(np.prod([i / patch_size for i in img_size]))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_size))
        self.patch_embed = Conv(
            in_channels=in_feats,
            out_channels=embed_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        x = x + self.pos_embed

        return x


class _TransformerLayer(nn.Module):
    def __init__(self, img_size, embed_size, num_heads, mlp_dim, Conv, Norm):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.msa = _MSA(embed_size, num_heads)

        self.norm2 = nn.LayerNorm(embed_size)
        self.mlp = _MLP(embed_size, num_heads, Norm)

    def forward(self, x):
        y = self.norm1(x)
        y = self.msa(y)
        x = x + y

        y = self.norm2(x)
        y = self.mlp(y)
        x = x + y

        return x


class _Transformer(nn.Sequential):
    def __init__(
        self,
        img_size,
        patch_size,
        in_feats,
        embed_size,
        num_heads,
        mlp_dim,
        num_layers,
        out_layers,
        Conv,
        Norm,
    ):
        super().__init__()
        self.out_layers = out_layers
        self.num_layers = num_layers
        self.embed = _Embedding(img_size, patch_size, in_feats, embed_size, Conv)
        self.layers = ModuleList([])

        for _ in range(0, num_layers):
            layer = _TransformerLayer(
                img_size, embed_size, num_heads, mlp_dim, Conv, Norm
            )
            self.layers.append(layer)

    def forward(self, x):
        out = []
        x = self.embed(x)

        for i in range(0, self.num_layers):
            x = self.layers[i](x)
            if i in self.out_layers:
                out.append(x)

        return out


def checkImgSize(img_size, number=4):
    if all([x >= 2**number for x in img_size]):
        return number
    else:
        return int(np.min(np.floor(np.log2(img_size))))


class unetr(ModelBase):
    """
    This is the U-NetR architecture : https://arxiv.org/abs/2103.10504.
    The 'residualConnections' flag controls residual connections The Downsampling, Encoding, Decoding modules are defined in the seg_modules file.
    These smaller modules are basically defined by 2 parameters, the input channels (filters) and the output channels (filters),
    and some other hyperparameters, which remain constant all the modules. For more details on the smaller modules please have a look at the seg_modules file.
    """

    def __init__(
        self,
        parameters: dict,
    ):
        super(unetr, self).__init__(parameters)

        # initialize defaults if not found
        parameters["model"]["inner_patch_size"] = parameters["model"].get(
            "inner_patch_size", parameters["patch_size"][0]
        )
        parameters["model"]["num_heads"] = parameters["model"].get("num_heads", 12)
        parameters["model"]["embed_dim"] = parameters["model"].get("embed_dim", 768)

        self.patch_size = parameters["model"]["inner_patch_size"]

        assert np.ceil(np.log2(parameters["model"]["inner_patch_size"])) == np.floor(
            np.log2(parameters["model"]["inner_patch_size"])
        ), "The inner patch size must be a power of 2."

        self.depth = int(np.log2(self.patch_size))
        parameters["model"]["depth"] = self.depth

        _ = self.model_depth_check(parameters)

        if self.n_dimensions == 2:
            self.img_size = parameters["patch_size"][0:2]
        elif self.n_dimensions == 3:
            self.img_size = parameters["patch_size"]

        self.num_layers = 3 * self.depth  # number of transformer layers
        self.out_layers = np.arange(2, self.num_layers, 3)

        self.num_heads = parameters["model"]["num_heads"]
        self.embed_size = parameters["model"]["embed_dim"]

        self.patch_dim = [i // self.patch_size for i in self.img_size]

        assert (
            self.embed_size % self.num_heads == 0
        ), "The embedding dimension must be divisible by the number of self-attention heads"

        assert all(
            [i % self.patch_size == 0 for i in self.img_size]
        ), "The image size is not divisible by the patch size in at least 1 dimension. UNETR is not defined in this case."

        assert all(
            [self.patch_size <= i for i in self.img_size]
        ), "The inner patch size must be smaller than the input image."

        self.transformer = _Transformer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_feats=self.n_channels,
            embed_size=self.embed_size,
            num_heads=self.num_heads,
            mlp_dim=2048,
            num_layers=self.num_layers,
            out_layers=self.out_layers,
            Conv=self.Conv,
            Norm=self.Norm,
        )

        self.upsampling = ModuleList([])
        self.convs = ModuleList([])

        for i in range(0, self.depth - 1):
            # add deconv blocks
            tempconvs = nn.Sequential()
            tempconvs.add_module(
                "conv0",
                _DeconvConvBlock(
                    self.embed_size,
                    32 * 2**self.depth,
                    self.Norm,
                    self.Conv,
                    self.ConvTranspose,
                ),
            )

            for j in range(self.depth - 2, i, -1):
                tempconvs.add_module(
                    "conv%d" % j,
                    _DeconvConvBlock(
                        128 * 2**j,
                        128 * 2 ** (j - 1),
                        self.Norm,
                        self.Conv,
                        self.ConvTranspose,
                    ),
                )

            self.convs.append(tempconvs)

            # add upsampling
            self.upsampling.append(
                _UpsampleBlock(
                    128 * 2 ** (i + 1), self.Norm, self.Conv, self.ConvTranspose
                )
            )

        # add upsampling for transformer output (no convs)
        self.upsampling.append(
            self.ConvTranspose(
                in_channels=self.embed_size,
                out_channels=32 * 2**self.depth,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            )
        )

        self.input_conv = nn.Sequential()
        self.input_conv.add_module(
            "conv1", _ConvBlock(self.n_channels, 32, self.Norm, self.Conv)
        )
        self.input_conv.add_module("conv2", _ConvBlock(32, 64, self.Norm, self.Conv))

        self.output_conv = nn.Sequential()
        self.output_conv.add_module("conv1", _ConvBlock(128, 64, self.Norm, self.Conv))
        self.output_conv.add_module("conv2", _ConvBlock(64, 64, self.Norm, self.Conv))
        self.output_conv.add_module(
            "conv3",
            out_conv(
                64,
                self.n_classes,
                conv_kwargs={
                    "kernel_size": 1,
                    "stride": 1,
                    "padding": 0,
                    "bias": False,
                },
                norm=self.Norm,
                conv=self.Conv,
                final_convolution_layer=self.final_convolution_layer,
                sigmoid_input_multiplier=self.sigmoid_input_multiplier,
            ),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Should be a 5D Tensor as [batch_size, channels, x_dims, y_dims, z_dims].

        Returns
        -------
        x : Tensor
            Returns a 5D Output Tensor as [batch_size, n_classes, x_dims, y_dims, z_dims].

        """
        transformer_out = self.transformer(x)

        y = self.upsampling[-1](
            transformer_out[-1]
            .transpose(-1, -2)
            .view(-1, self.embed_size, *self.patch_dim)
        )  # z12

        for i in range(len(self.convs) - 1, -1, -1):
            zi = (
                transformer_out[i]
                .transpose(-1, -2)
                .view(-1, self.embed_size, *self.patch_dim)
            )
            zi = self.convs[i](zi)
            zicat = torch.cat([zi, y], dim=1)
            y = self.upsampling[i](zicat)

        x = self.input_conv(x)
        x = torch.cat([x, y], dim=1)
        x = self.output_conv(x)

        return x
