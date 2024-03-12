"""Implementation of DCGAN model."""

import torch
import torch.nn as nn
from GANDLF.models.modelBase import ModelBase
from warnings import warn
from typing import Dict, Tuple


class _GeneratorDCGAN(nn.Module):
    """Generator for the DCGAN."""

    def __init__(
        self,
        output_patch_size: Tuple[int, int, int],
        n_dimensions: int,
        latent_vector_dim: int,
        num_output_channels: int,
        growth_rate: int,
        gen_init_channels: int,
        norm: nn.Module,
        conv: nn.Module,
    ) -> None:
        """
        Initializes a new instance of the _GneratorDCGAN class.
        Parameters:
        output_patch_size (Tuple[int, int,int]): The size of the output
        patch.
            n_dimensions (int): The dimensionality of the input and output.
            latent_vector_dim (int): The dimension of the latent vector
        to be used as input to the generator.
            num_output_channels (int): The number of output channels in
        the generated image.
            growth_rate (int): The growth rate of the number of hidden
        features in the consecutive layers of the generator. Note that
        in this case the growth will be in reverse order, i.e. the number
        of channels will DECREASE by this amount.
            gen_init_channels (int): Initial number of channels in the
        generator, which is scaled by the growth rate in the subsequent
        layers.
            norm (torch.nn.module): A normalization layer subclassing
        torch.nn.Module (i.e. nn.BatchNorm2d)
            conv (torch.nn.module): A convolutional layer subclassing
        torch.nn.Module. Note that in this case this
        should be a transposed convolution (i.e. nn.ConvTranspose2d).
        """
        super().__init__()
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module(
            "conv1t",
            conv(latent_vector_dim, gen_init_channels, 4, 1, 0, bias=False),
        )
        self.feature_extractor.add_module("norm1", norm(gen_init_channels))
        self.feature_extractor.add_module("relu1", nn.ReLU(inplace=True))
        self.feature_extractor.add_module(
            "conv2t",
            conv(
                gen_init_channels,
                gen_init_channels // growth_rate,
                4,
                2,
                1,
                bias=False,
            ),
        )
        self.feature_extractor.add_module(
            "norm2", norm(gen_init_channels // growth_rate)
        )
        self.feature_extractor.add_module("relu2", nn.ReLU(inplace=True))
        self.feature_extractor.add_module(
            "conv3t",
            conv(
                gen_init_channels // growth_rate,
                gen_init_channels // (growth_rate**2),
                4,
                2,
                1,
                bias=False,
            ),
        )
        self.feature_extractor.add_module(
            "norm3", norm(gen_init_channels // (growth_rate**2))
        )
        self.feature_extractor.add_module("relu3", nn.ReLU(inplace=True))
        self.feature_extractor.add_module(
            "conv4t",
            conv(
                gen_init_channels // (growth_rate**2),
                gen_init_channels // (growth_rate**3),
                4,
                2,
                1,
                bias=False,
            ),
        )
        self.feature_extractor.add_module(
            "norm4", norm(gen_init_channels // (growth_rate**3))
        )
        self.feature_extractor.add_module("relu4", nn.ReLU(inplace=True))

        self.feature_extractor.add_module(
            "conv5t",
            conv(
                gen_init_channels // (growth_rate**3),
                num_output_channels,
                4,
                2,
                1,
                bias=False,
            ),
        )
        feature_extractor_output_size = (
            self._get_output_size_feature_extractor(
                self.feature_extractor, latent_vector_dim, n_dimensions
            )
        )
        # if the output size of the feature extractor does not match
        # the output patch size, add an upsampling layer and a 1x1
        # convolution to match the output size and reparametrize the
        # interpoladed output
        if not self._output_shape_matching(
            output_patch_size, feature_extractor_output_size
        ):
            self.feature_extractor.add_module(
                "upsample",
                nn.Upsample(
                    size=(
                        output_patch_size[:-1]
                        if n_dimensions == 2
                        else output_patch_size
                    ),
                    mode="bilinear" if n_dimensions == 2 else "trilinear",
                    align_corners=True,
                ),
            )
            self.feature_extractor.add_module(
                "conv5",
                conv(
                    num_output_channels, num_output_channels, 1, 1, bias=False
                ),
            )

        self.feature_extractor.add_module("tanh", nn.Tanh())

    @staticmethod
    def _output_shape_matching(
        output_patch_size: Tuple[int, int, int],
        feature_extractor_output_size: Tuple[int, int, int],
    ) -> bool:
        """Checks if the output patch size matches the output size of
        the feature extractor.
        Args:
            output_patch_size (Tuple[int, int, int]): The size of the
        output patch.
            feature_extractor_output_size (Tuple[int, int, int]): The
        output size of the feature extractor.
        Returns:
            bool: True if the output patch size matches the output size
        """
        if output_patch_size[-1] == 1:
            output_patch_size = output_patch_size[:-1]
        if output_patch_size != feature_extractor_output_size:
            return False
        return True

    @staticmethod
    def _get_output_size_feature_extractor(
        feature_extractor: nn.Module,
        latent_vector_dim: int,
        n_dimensions: int = 3,
    ) -> int:
        """Determines the output size of the feature extractor to
        initialize the classifier.
        Args:
            feature_extractor (nn.Module): The feature extractor module.
            latent_vector_dim (int): The dimension of the latent vector
        to be used as input to the generator.
        Returns:
            int: The output size of the feature extractor.
        """
        dummy_input_shape = [1, latent_vector_dim, 1, 1]
        if n_dimensions == 3:
            dummy_input_shape.append(1)
        dummy_input = torch.randn(dummy_input_shape)
        dummy_output = feature_extractor(dummy_input)
        return dummy_output.shape[2:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        Parameters:
            x (torch.Tensor): The latent vector to be used as input to
        the generator.
        Returns:
            torch.Tensor: The generated image.
        """
        out = self.feature_extractor(x)
        return out


class _DiscriminatorDCGAN(nn.Module):
    """Discriminator for the DCGAN."""

    def __init__(
        self,
        input_patch_size: Tuple[int, int, int],
        n_dimensions: int,
        num_input_channels: int,
        growth_rate: int,
        disc_init_channels: int,
        slope: float,
        norm: nn.Module,
        conv: nn.Module,
    ) -> None:
        """
        Initializes a new instance of the _DiscriminatorDCGAN class.
        Parameters:
            input_patch_size (Tuple[int, int,int]): The size of the
        input patch.
            n_dimensions (int): The dimensionality of the input.
            num_input_channels (int): The number of input channels in
        the image to be discriminated.
            growth_rate (int): The growth rate of the number of hidden
        features in the consecutive layers of the discriminator.
            disc_init_channels (int): Initial number of channels in the
        discriminator, which is scaled by the growth rate in the subsequent
        layers.
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
            conv(num_input_channels, disc_init_channels, 4, 2, 1, bias=False),
        )
        self.feature_extractor.add_module(
            "leaky_relu1", nn.LeakyReLU(slope, inplace=False)
        )
        self.feature_extractor.add_module(
            "conv2",
            conv(
                disc_init_channels,
                disc_init_channels * growth_rate,
                3,
                1,
                1,
                bias=False,
            ),
        )
        self.feature_extractor.add_module(
            "norm2", norm(disc_init_channels * growth_rate)
        )
        self.feature_extractor.add_module(
            "leaky_relu2", nn.LeakyReLU(slope, inplace=False)
        )
        self.feature_extractor.add_module(
            "conv3",
            conv(
                disc_init_channels * growth_rate,
                disc_init_channels * (growth_rate**2),
                3,
                1,
                1,
                bias=False,
            ),
        )
        self.feature_extractor.add_module(
            "norm3", norm(disc_init_channels * (growth_rate**2))
        )
        self.feature_extractor.add_module(
            "leaky_relu3", nn.LeakyReLU(slope, inplace=False)
        )
        self.feature_extractor.add_module(
            "conv4",
            conv(
                disc_init_channels * (growth_rate**2),
                disc_init_channels * (growth_rate**3),
                3,
                2,
                1,
                bias=False,
            ),
        )
        self.feature_extractor.add_module(
            "norm4", norm(disc_init_channels * (growth_rate**3))
        )
        self.feature_extractor.add_module(
            "leaky_relu4", nn.LeakyReLU(slope, inplace=False)
        )

        self.feature_extractor.add_module(
            "conv5",
            conv(
                disc_init_channels * (growth_rate**3),
                1,
                3,
                1,
                0,
                bias=False,
            ),
        )
        self.feature_extractor.add_module("flatten", nn.Flatten(start_dim=1))

        num_output_features = self._get_output_size_feature_extractor(
            self.feature_extractor,
            input_patch_size,
            num_input_channels,
            n_dimensions,
        )
        self.classifier.add_module(
            "linear1", nn.Linear(num_output_features, 1)
        )
        self.classifier.add_module("sigmoid", nn.Sigmoid())

    @staticmethod
    def _get_output_size_feature_extractor(
        feature_extractor: nn.Module,
        patch_size: Tuple[int, int, int],
        n_channels: int = 1,
        n_dimensions: int = 3,
    ) -> int:
        """Determines the output size of the feature extractor to
        initialize the classifier.
        Args:
            feature_extractor (nn.Module): The feature extractor module.
            patch_size (Tuple[int, int,int]): The size of the input patch.
            n_channels (int): The number of input channels in the image
        to be discriminated.
            n_dimensions (int): The dimensionality of the input.
        Returns:
            int: The output size of the feature extractor.
        """
        dummy_input_shape = [1, n_channels, *patch_size]
        if n_dimensions == 2:
            dummy_input_shape.pop()
        dummy_input = torch.randn(dummy_input_shape)
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
        if not ("latent_vector_size" in parameters["model"]):
            warn(
                "No latent vector dimension specified. Defaulting to 100.",
                RuntimeWarning,
            )
            parameters["model"]["latent_vector_size"] = 100
        if not ("growth_rate_gen" in parameters["model"]):
            parameters["model"]["growth_rate_gen"] = 2
        if not ("init_channels_gen" in parameters["model"]):
            parameters["model"]["init_channels_gen"] = 512
        if not ("growth_rate_disc" in parameters["model"]):
            parameters["model"]["growth_rate_disc"] = 2
        if not ("init_channels_disc" in parameters["model"]):
            parameters["model"]["init_channels_disc"] = 64
        if not ("slope" in parameters["model"]):
            parameters["model"]["slope"] = 0.2
        if self.Norm is None:
            warn(
                "No normalization specified. Defaulting to BatchNorm",
                RuntimeWarning,
            )
            self.Norm = self.BatchNorm
        self.generator = _GeneratorDCGAN(
            output_patch_size=self.patch_size,
            n_dimensions=self.n_dimensions,
            latent_vector_dim=parameters["model"]["latent_vector_size"],
            num_output_channels=self.n_channels,
            growth_rate=parameters["model"]["growth_rate_gen"],
            gen_init_channels=parameters["model"]["init_channels_gen"],
            norm=self.Norm,
            conv=self.ConvTranspose,
        )
        self.discriminator = _DiscriminatorDCGAN(
            input_patch_size=self.patch_size,
            n_dimensions=self.n_dimensions,
            num_input_channels=self.n_channels,
            growth_rate=parameters["model"]["growth_rate_disc"],
            disc_init_channels=parameters["model"]["init_channels_disc"],
            slope=parameters["model"]["slope"],
            norm=self.Norm,
            conv=self.Conv,
        )
        # TODO this initialization is preventing the model from convergence
        # self._init_generator_weights(self.generator)
        # self._init_discriminator_weights(self.discriminator)

    def _init_generator_weights(self, generator: nn.Module) -> None:
        """
        Initializes the weights of the generator. This is mimicking the
        original implementation of the DCGAN.
        Parameters:
            generator (torch.nn.Module): The generator module.
        """
        for m in generator.modules():
            if isinstance(m, self.ConvTranspose):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, self.Norm):
                if m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def _init_discriminator_weights(self, discriminator: nn.Module) -> None:
        """
        Initializes the weights of the discriminator. This is mimicking the
        original implementation of the DCGAN.
        Parameters:
            discriminator (torch.nn.Module): The discriminator module.
        """
        for m in discriminator.modules():
            if isinstance(m, self.ConvTranspose):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, self.Norm):
                if m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass, implemented simply as generator_forward.
        Parameters:
            x (torch.Tensor): The latent vector to be used as input to
        the generator.
        Returns:
            torch.Tensor: The generated image.
        """
        return self.generator_forward(x)
