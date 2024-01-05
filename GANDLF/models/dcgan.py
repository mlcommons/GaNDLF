"""Implementation of DCGAN model."""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelBase import ModelBase
from warnings import warn
from typing import Dict, Tuple, Union
from GANDLF.parseConfig import parseConfig


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
        super().__init__()
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


class _DiscriminatorDCGAN(nn.Module):
    """Discriminator for the DCGAN."""

    def __init__(
        self,
        input_patch_size: Union[Tuple[int, int, int], Tuple[int, int]],
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        slope: float,
        norm: nn.Module,
        conv: nn.Module,
    ) -> None:
        """
        Initializes a new instance of the _DiscriminatorDCGAN class.
        Parameters:
            num_input_features (int): The number of input channels in
        the image to be discriminated.
            growth_rate (int): The growth rate of the number of hidden
        features in the consecutive layers of the discriminator.
            bn_size (int): Factor to scale the number of intermediate
        channels between the 1x1 and 3x3 convolutions.
            drop_rate (float): The dropout rate in the classifier.
            slope (float): The slope of the LeakyReLU activation function.
            norm (torch.nn.module): A normalization layer subclassing
        torch.nn.Module (i.e. nn.BatchNorm2d)
            conv (torch.nn.module): A convolutional layer subclassing
        torch.nn.Module.
        """
        super().__init__()
        self.feature_extractor = nn.Sequential()
        self.classifier = nn.Sequential()
        self.feature_extractor.add_module(
            "conv1",
            conv(num_input_features, bn_size, 4, 2, 1, bias=False),
        )
        self.feature_extractor.add_module(
            "leaky_relu1", nn.LeakyReLU(slope, inplace=True)
        )
        self.feature_extractor.add_module(
            "conv2",
            conv(bn_size, bn_size * growth_rate, 4, 2, 1, bias=False),
        )
        self.feature_extractor.add_module("norm2", norm(bn_size * growth_rate))
        self.feature_extractor.add_module(
            "leaky_relu2", nn.LeakyReLU(slope, inplace=True)
        )
        self.feature_extractor.add_module(
            "conv3",
            conv(
                bn_size * growth_rate,
                bn_size * (growth_rate**2),
                4,
                2,
                1,
                bias=False,
            ),
        )
        self.feature_extractor.add_module(
            "norm3", norm(bn_size * (growth_rate**2))
        )
        self.feature_extractor.add_module(
            "leaky_relu3", nn.LeakyReLU(slope, inplace=True)
        )
        self.feature_extractor.add_module(
            "conv4",
            conv(
                bn_size * (growth_rate**2),
                1,
                4,
                1,
                0,
                bias=False,
            ),
        )
        self.feature_extractor.add_module("flatten", nn.Flatten(start_dim=1))

        num_output_features = self._get_output_size_feature_extractor(
            self.feature_extractor, input_patch_size
        )
        self.classifier.add_module(
            "linear1", nn.Linear(num_output_features, 128)
        )
        self.classifier.add_module("dropout1", nn.Dropout(drop_rate))
        self.classifier.add_module(
            "leaky_relu1", nn.LeakyReLU(slope, inplace=True)
        )
        self.classifier.add_module("linear2", nn.Linear(128, 1))
        self.classifier.add_module("sigmoid", nn.Sigmoid())

    @staticmethod
    def _get_output_size_feature_extractor(
        feature_extractor: nn.Module,
        patch_size: Union[Tuple[int, int, int], Tuple[int, int]],
        n_channels: int = 1,
    ) -> int:
        """Determines the output size of the feature extractor to
        initialize the classifier.
        """
        dummy_input = torch.randn((1, n_channels, *patch_size[:-1]))
        dummy_output = feature_extractor(dummy_input)
        return dummy_output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        Parameters:
            x (torch.Tensor): The image to be discriminated.
        Returns:
            torch.Tensor: The probability that the image is real.
        """
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out


class DCGAN(ModelBase):
    """
    DCGAN model class. This class implements the architecture and forward
    passes for the generator and discriminator subnetworks of the DCGAN.
    """

    def __init__(self, parameters: Dict):
        ModelBase.__init__(self, parameters)
        if not ("latent_vector_dim" in parameters):
            warn(
                "No latent vector dimension specified. Defaulting to 100.",
                RuntimeWarning,
            )
            parameters["latent_vector_dim"] = 100
        if not ("growth_rate" in parameters):
            parameters["growth_rate"] = 2
        if not ("bn_size" in parameters):
            parameters["bn_size"] = 4
        if not ("slope" in parameters):
            parameters["slope"] = 0.2
        if not ("drop_rate" in parameters):
            parameters["drop_rate"] = 0.0
        if not ("conv1_t_stride" in parameters):
            parameters["conv1_t_stride"] = 1
        if not ("conv1_t_size" in parameters):
            parameters["conv1_t_size"] = 7
        if self.Norm is None:
            warn(
                "No normalization specified. Defaulting to BatchNorm",
                RuntimeWarning,
            )
            self.Norm = self.BatchNorm
        self.generator = _GneratorDCGAN(
            parameters["latent_vector_dim"],
            self.n_channels,
            parameters["growth_rate"],
            parameters["bn_size"],
            parameters["slope"],
            self.Norm,
            self.ConvTranspose,
        )
        self.discriminator = _DiscriminatorDCGAN(
            self.patch_size,
            self.n_channels,
            parameters["growth_rate"],
            parameters["bn_size"],
            parameters["drop_rate"],
            parameters["slope"],
            self.Norm,
            self.Conv,
        )
        self._init_generator_weights(self.generator, parameters["slope"])
        self._init_discriminator_weights(
            self.discriminator, parameters["slope"]
        )

    def _init_generator_weights(
        self, generator: nn.Module, leaky_relu_slope: float
    ) -> None:
        """
        Initializes the weights of the generator.
        Parameters:
            m (torch.nn.Module): The generator module.
        """
        for m in generator.modules():
            if isinstance(m, self.ConvTranspose):
                nn.init.kaiming_normal_(
                    m.weight, a=leaky_relu_slope, mode="fan_out"
                )
            elif isinstance(m, self.Norm):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _init_discriminator_weights(
        self, discriminator: nn.Module, leaky_relu_slope: float
    ) -> None:
        """
        Initializes the weights of the discriminator.
        Parameters:
            m (torch.nn.Module): The discriminator module.
        """
        for m in discriminator.modules():
            if isinstance(m, self.Conv):
                nn.init.kaiming_normal_(
                    m.weight, a=leaky_relu_slope, mode="fan_in"
                )
            elif isinstance(m, self.Norm):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def generator_forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        Parameters:
            latent_vector (torch.Tensor): The latent vector to be used as
        input to the generator.
        Returns:
            torch.Tensor: The generated image.
        """
        return self.generator(latent_vector)

    def discriminator_forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        Parameters:
            image (torch.Tensor): The image to be discriminated.
        Returns:
            torch.Tensor: The probability that the image is real.
        """
        return self.discriminator(image)


if __name__ == "__main__":
    testingDir = "/home/szymon/code/GaNDLF/testing"
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )

    parameters["model"]["architecture"] = "DCGAN"
    parameters["model"]["dimension"] = 2
    parameters["model"]["num_channels"] = 1
    patch_size = parameters["patch_size"]
    fake_input = torch.randn((1, 100, 1, 1))
    real_image = torch.randn((4, 1, *patch_size[:-1]))
    model = DCGAN(parameters)
    fake_output = model.generator_forward(fake_input)
    disc_output = model.discriminator_forward(real_image)
