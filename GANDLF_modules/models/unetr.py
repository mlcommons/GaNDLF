from .modelBase import ModelBase
from GANDLF.models.seg_modules.out_conv import out_conv
import torch
import torch.nn as nn
from torch.nn import ModuleList
import numpy as np
import math


class _DeconvConvBlock(nn.Sequential):
    """
    A block consisting of a transposed convolutional layer, followed by a convolutional layer,
    a normalization layer, and a ReLU activation function. The block is defined as a sequential
    module in PyTorch, making it easy to stack multiple blocks together.

    Args:
        in_feats (int): The number of input features to the block.
        out_feats (int): The number of output features from the block.
        Norm (torch.nn.Module): The normalization layer to use (e.g. BatchNorm2d).
        Conv (torch.nn.Module): The convolutional layer to use (e.g. Conv2d).
        Deconv (torch.nn.Module): The transposed convolutional layer to use (e.g. ConvTranspose2d).
    """

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
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            x (torch.Tensor): The output tensor.
        """

        # Apply deconv -> conv -> norm -> relu
        x = self.deconv(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class _ConvBlock(nn.Sequential):
    """
    A block consisting of a convolutional layer followed by batch normalization and ReLU activation.

    Args:
        in_feats (int): Number of input features.
        out_feats (int): Number of output features.
        Norm (nn.Module): A normalization layer.
        Conv (nn.Module): A convolutional layer.
    """

    def __init__(self, in_feats, out_feats, Norm, Conv):
        """
        Initializes the ConvBlock.

        Args:
            in_feats (int): Number of input features.
            out_feats (int): Number of output features.
            Norm (nn.Module): A normalization layer.
            Conv (nn.Module): A convolutional layer.
        """
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
        """
        Performs a forward pass of the ConvBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_feats, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_feats, height, width).
        """

        # Apply conv -> norm -> relu
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class _UpsampleBlock(nn.Sequential):
    def __init__(self, in_feats, Norm, Conv, Deconv):
        """
        Args:
            in_feats (int): Number of input channels.
            Norm (nn.Module): Normalization layer constructor.
            Conv (nn.Module): Convolutional layer constructor.
            Deconv (nn.Module): Transposed convolutional layer constructor.
        """
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
        """
        Performs a forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_feats, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_feats//4, 2*height, 2*width).
        """

        # Apply conv1 -> norm1 -> relu1-> relu2
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # Apply conv2 -> norm2 -> relu2
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        # Apply deconv
        x = self.deconv(x)

        return x


class _MLP(nn.Sequential):
    """
    Multi-layer perceptron module that applies two linear transformations with GELU activations.

    Args:
        in_feats (int): Number of input features.
        out_feats (int): Number of output features for the first linear transformation.
        Norm (nn.Module): Normalization module, e.g. LayerNorm.

    """

    def __init__(self, in_feats, out_feats):  # Which normalization module to use?
        super().__init__()

        self.add_module("norm", nn.LayerNorm(in_feats))

        self.add_module("linear1", nn.Linear(in_feats, out_feats))
        self.add_module("gelu1", nn.GELU())

        self.add_module("linear2", nn.Linear(out_feats, in_feats))
        self.add_module("gelu2", nn.GELU())

    def forward(self, x):
        """
        Applies the MLP module to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_feats).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, in_feats).
        """

        # Apply norm -> linear1 -> gelu1
        x = self.norm(x)
        x = self.linear1(x)
        x = self.gelu1(x)

        # Apply linear2 -> gelu2
        x = self.linear2(x)
        x = self.gelu2(x)

        return x


class _MSA(nn.Module):
    """
    Multi-Head Self-Attention module.

    Args:
        embed_size (int): The size of the input feature embedding.
        num_heads (int): The number of attention heads to use.

    Attributes:
        num_heads (int): The number of attention heads used.
        query_size (int): The size of each query vector.
        all_heads (int): The total size of all query, key, and value vectors.

        query (nn.Linear): Linear layer to project input features into query vectors.
        key (nn.Linear): Linear layer to project input features into key vectors.
        value (nn.Linear): Linear layer to project input features into value vectors.
        out (nn.Linear): Linear layer to project concatenated attention head outputs.

        softmax (nn.Softmax): Softmax function used to compute attention weights.

    Methods:
        reshape(x): Reshapes the input tensor to enable batch-wise matrix multiplication.
        forward(x): Computes multi-head self-attention on input tensor.

    Example:
        >>> msa = _MSA(embed_size=256, num_heads=8)
        >>> x = torch.randn(4, 16, 256)  # batch size = 4, sequence length = 16, embedding size = 256
        >>> output = msa(x)  # output shape: (4, 16, 256)
    """

    def __init__(self, embed_size, num_heads):
        super().__init__()

        # Initialize parameters
        self.num_heads = num_heads
        self.query_size = int(embed_size / self.num_heads)
        self.all_heads = self.query_size * self.num_heads  # should equal embed_size

        # Initialize layers
        self.query = nn.Linear(embed_size, self.all_heads)
        self.key = nn.Linear(embed_size, self.all_heads)
        self.value = nn.Linear(embed_size, self.all_heads)

        self.out = nn.Linear(self.all_heads, self.all_heads)

        self.softmax = nn.Softmax(dim=-1)

    def reshape(self, x):
        """
        Reshapes the input tensor to enable batch-wise matrix multiplication.

        Args:
            x (torch.Tensor): The input tensor to reshape.

        Returns:
            The reshaped tensor.
        """
        x_shape = list(x.size()[:-1]) + [self.num_heads, self.query_size]
        x = x.view(*x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        """
        Computes multi-head self-attention on input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape `(batch_size, sequence_length, embed_size)`.

        Returns:
            The output tensor of shape `(batch_size, sequence_length, embed_size)`.
        """
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
        """
        A module that creates embeddings from an image tensor.

        Args:
            img_size (tuple[int]): The size of the input image tensor (H, W, C).
            patch_size (int): The size of each square patch that the image is divided into.
            in_feats (int): The number of input channels in the image tensor.
            embed_size (int): The size of the output embedding vector.
            Conv (nn.Module): The convolutional module to use for extracting patches.
        """
        super().__init__()

        # Calculate the number of patches in the image
        self.n_patches = int(np.prod([i / patch_size for i in img_size]))

        # Create a learnable parameter for the positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_size))

        # Create a convolutional module to extract patches from the input image
        self.patch_embed = Conv(
            in_channels=in_feats,
            out_channels=embed_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """
        Forward pass for the _Embedding module.

        Args:
            x (torch.Tensor): An input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: An output tensor of shape (B, n_patches, embed_size).
        """
        # Extract patches from the input image
        x = self.patch_embed(x)

        # Flatten the patches into a 2D matrix
        x = x.flatten(2)

        # Transpose the matrix so that patches are in the first dimension
        x = x.transpose(-1, -2)

        # Add the positional embedding to the flattened patches
        x = x + self.pos_embed

        return x


class _TransformerLayer(nn.Module):
    def __init__(self, embed_size, num_heads):
        """
        A module that implements a single transformer layer.

        Args:
            embed_size (int): The size of the embedding vector.
            num_heads (int): The number of heads to use in the multi-head self-attention module.
        """
        super().__init__()

        # Create normalization modules and a multi-head self-attention module
        self.norm1 = nn.LayerNorm(embed_size)
        self.msa = _MSA(embed_size, num_heads)

        # Create normalization modules and an MLP
        self.norm2 = nn.LayerNorm(embed_size)

        # Dev note: it should be out_feats=mlp_dim, but we have out_feats=num_heads
        # Also, the Norm parameter is not used in the MLP module
        # So this needs to be looked at later, but for now, it works
        self.mlp = _MLP(in_feats=embed_size, out_feats=num_heads)

    def forward(self, x):
        """
        Forward pass for the _TransformerLayer module.

        Args:
            x (torch.Tensor): An input tensor of shape (B, n_patches, embed_size).

        Returns:
            torch.Tensor: An output tensor of the same shape as the input.
        """
        # Normalize the input and apply multi-head self-attention
        y = self.norm1(x)
        y = self.msa(y)

        # Add the input to the output of the self-attention module
        x = x + y

        # Normalize the output of the self
        y = self.norm2(x)
        y = self.mlp(y)

        # Add the input to the output of the MLP
        x = x + y

        return x


class _Transformer(nn.Sequential):
    """
    A transformer module that consists of an embedding layer followed by a series of transformer layers.

    Parameters:
        img_size (tuple): The dimensions of the input image (height, width, depth).
        patch_size (int): The size of the patches to be extracted from the input image.
        in_feats (int): The number of input features.
        embed_size (int): The size of the embedding.
        num_heads (int): The number of attention heads to use in the multi-head attention layer.
        num_layers (int): The number of transformer layers to use.
        out_layers (list): A list of indices indicating which transformer layers should output their results.
        Norm (nn.Module): A normalization module to use for the transformer layers.

    Attributes:
        out_layers (list): A list of indices indicating which transformer layers should output their results.
        num_layers (int): The number of transformer layers to use.
        embed (_Embedding): The embedding layer.
        layers (ModuleList): A list of _TransformerLayer objects.

    Methods:
        forward(x): Processes the input through the transformer and returns the output.

    Example:
        transformer = _Transformer(
            img_size=(224, 224, 3),
            patch_size=16,
            in_feats=3,
            embed_size=256,
            num_heads=8,
            mlp_dim=2048,
            num_layers=12,
            out_layers=[5, 11],
            Conv=nn.Conv2d,
            Norm=nn.LayerNorm
        )
        output = transformer(input_tensor)
    """

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
                embed_size,
                num_heads,
            )
            self.layers.append(layer)

    def forward(self, x):
        """
        Processes the input through the transformer and returns the output.

        Parameters:
            x (tensor): The input tensor.

        Returns:
            out (list): A list of tensors representing the output of the specified transformer layers.
        """
        out = []
        x = self.embed(x)

        # Note : Lists are highly inefficient when dealing with tensor operations and must be used carefully
        for i in range(0, self.num_layers):
            x = self.layers[i](x)
            if i in self.out_layers:
                out.append(x)

        return out


def checkImgSize(img_size, number=4):
    """
    Checks if the input image size is greater than or equal to 2^number in each dimension.
    If it is, it returns the specified number. Otherwise, it returns the minimum log base 2
    of the input image size.

    Args:
        img_size (tuple of ints): size of the input image tensor
        number (int): a number that the input image size is checked against. Default value is 4.

    Returns:
        int : An integer, either the specified number or the minimum log base 2 of the input image size.
    """
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
        """
        Initializes an instance of the `unetr` class.

        Args:
            parameters (dict): A dictionary containing the model parameters.

        Raises:
        -------
        AssertionError
            If the input image size is not divisible by the patch size in at least 1 dimension, or if the inner patch size is not smaller than the input image.
            If the embedding dimension is not divisible by the number of self-attention heads.

        """
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

        # Dev note : Need to update the hard-coded number of channel modules here
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
        Perform the forward pass of the UNet model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape [batch_size, channels, x_dims, y_dims, z_dims].

        Returns
        -------
        torch.Tensor
            The output tensor of shape [batch_size, n_classes, x_dims, y_dims, z_dims].
        """

        # Perform transformer encoding of input tensor
        transformer_out = self.transformer(x)

        # Perform upsampling on last transformer output and concatenate with previous outputs
        y = self.upsampling[-1](
            transformer_out[-1]
            .transpose(-1, -2)
            .view(-1, self.embed_size, *self.patch_dim)
        )  # z12

        for i in range(len(self.convs) - 1, -1, -1):
            # Perform convolution on transformer output and concatenate with previous outputs
            zi = (
                transformer_out[i]
                .transpose(-1, -2)
                .view(-1, self.embed_size, *self.patch_dim)
            )

            zi = self.convs[i](zi)

            # Perform convolution on concatenated output
            zicat = torch.cat([zi, y], dim=1)

            # Perform upsampling on concatenated output
            y = self.upsampling[i](zicat)

        # Perform convolution on input tensor and concatenate with final output
        x = self.input_conv(x)
        x = torch.cat([x, y], dim=1)
        x = self.output_conv(x)

        return x
