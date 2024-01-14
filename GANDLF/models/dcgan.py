"""Implementation of DCGAN model."""
import torch
import torch.nn as nn
from GANDLF.models.modelBase import ModelBase
from warnings import warn
from typing import Dict, Tuple


class _GneratorDCGAN(nn.Module):
    """Generator for the DCGAN."""

    def __init__(
        self,
        output_patch_size: Tuple[int, int, int],
        n_dimensions: int,
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
        output_patch_size (Tuple[int, int,int]): The size of the output
        patch.
            n_dimensions (int): The dimensionality of the input and output.
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
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module(
            "conv1t",
            conv(latent_vector_dim, bn_size, 4, 1, 0, bias=False),
        )
        self.feature_extractor.add_module("norm1", norm(bn_size))
        self.feature_extractor.add_module(
            "leaky_relu1", nn.LeakyReLU(slope, inplace=True)
        )
        self.feature_extractor.add_module(
            "conv2t",
            conv(bn_size, bn_size // growth_rate, 4, 2, 1, bias=False),
        )
        self.feature_extractor.add_module(
            "norm2", norm(bn_size // growth_rate)
        )
        self.feature_extractor.add_module(
            "leaky_relu2", nn.LeakyReLU(slope, inplace=True)
        )
        self.feature_extractor.add_module(
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
        self.feature_extractor.add_module(
            "norm3", norm(bn_size // (growth_rate**2))
        )
        self.feature_extractor.add_module(
            "leaky_relu3", nn.LeakyReLU(slope, inplace=True)
        )
        self.feature_extractor.add_module(
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
                    size=output_patch_size[:-1]
                    if n_dimensions == 2
                    else output_patch_size,
                    mode="bilinear" if n_dimensions == 2 else "trilinear",
                    align_corners=True,
                ),
            )
            self.feature_extractor.add_module(
                "conv5",
                conv(
                    num_output_features, num_output_features, 1, 1, bias=False
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
            input_patch_size (Tuple[int, int,int]): The size of the
        input patch.
            n_dimensions (int): The dimensionality of the input.
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
            self.feature_extractor,
            input_patch_size,
            num_input_features,
            n_dimensions,
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
            self.patch_size,
            self.n_dimensions,
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
            self.n_dimensions,
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
