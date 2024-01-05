"""Implementation of DCGAN model."""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class _GneratorDCGAN(nn.Sequential):
    """Generator for the DCGAN."""

    def __init__(
        self,
        latent_vector_dim: int,
        num_output_features: int,
        growth_rate: int,
        bn_size: int,
        slope: float,
        norm: nn.Module,
        conv: nn.Module,
    ) -> None:
        """
        Initializes a new instance of the _GneratorDCGAN class.
        Parameters:
            latent_vector_dim (int): The dimension of the latent vector
        to be used as input to the generator.
            num_output_features (int): The number of output channels in
        the generated image.
            growth_rate (int): The growth rate of the number of hidden
        features in the consecutive layers of the generator. Note that
        in this case the growth will be in reverse order, i.e. the number
        of channels will DECREASE by this amount.
            bn_size (int): Factor to scale the number of intermediate
        channels between the 1x1 and 3x3 convolutions.
            slope (float): The slope of the LeakyReLU activation function.
            norm (torch.nn.module): A normalization layer subclassing
        torch.nn.Module (i.e. nn.BatchNorm2d)
            conv (torch.nn.module): A convolutional layer subclassing
        torch.nn.Module. Note that in this case this
        should be a transposed convolution (i.e. nn.ConvTranspose2d).
        """
        super.__init__()
        self.add_module(
            "conv1t",
            conv(latent_vector_dim, bn_size, 4, 1, 0, bias=False),
        )
        self.add_module("norm1", norm(bn_size))
        self.add_module("leaky_relu1", nn.LeakyReLU(slope, inplace=True))
        self.add_module(
            "conv2t",
            conv(bn_size, bn_size // growth_rate, 4, 2, 1, bias=False),
        )
        self.add_module("norm2", norm(bn_size // growth_rate))
        self.add_module("leaky_relu2", nn.LeakyReLU(slope, inplace=True))
        self.add_module(
            "conv3t",
            conv(
                bn_size // growth_rate,
                bn_size // (growth_rate**2),
                4,
                2,
                1,
                bias=False,
            ),
        )
        self.add_module("norm3", norm(bn_size // (growth_rate**2)))
        self.add_module("leaky_relu3", nn.LeakyReLU(slope, inplace=True))
        self.add_module(
            "conv4t",
            conv(
                bn_size // (growth_rate**2),
                num_output_features,
                4,
                2,
                1,
                bias=False,
            ),
        )
        self.add_module("tanh", nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        Parameters:
            x (torch.Tensor): The latent vector to be used as input to
        the generator.
        Returns:
            torch.Tensor: The generated image.
        """
        return super().forward(x)
