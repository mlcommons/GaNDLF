from typing import Tuple, Union
import numpy as np
import torch
from skimage.color import rgb2hed, hed2rgb
from torchio.transforms.augmentation import RandomTransform
from torchio.transforms import IntensityTransform
from torchio import Subject


def hed_transform(parameters):
    return RandomHEDTransform(
        haematoxylin_sigma_range=parameters["haematoxylin_sigma_range"],
        haematoxylin_bias_range=parameters["haematoxylin_bias_range"],
        eosin_sigma_range=parameters["eosin_sigma_range"],
        eosin_bias_range=parameters["eosin_bias_range"],
        dab_sigma_range=parameters["dab_sigma_range"],
        dab_bias_range=parameters["dab_bias_range"],
        cutoff_range=parameters["cutoff_range"],
    )


class AugmenterBase:
    """Base class for patch augmentation with a hed transform"""

    def __init__(self, keyword):
        """
        Args:
            keyword (str): Short name for the transformation.
        """
        self._keyword = keyword

    ## commented the following lines because the user is never given access to these
    # @property
    # def keyword(self):
    #     """Get the keyword for the augmenter."""
    #     return self._keyword

    # def shapes(self, target_shapes):
    #     """Calculate the required shape of the input to achieve the target output shape."""
    #     return target_shapes

    # def transform(self, patch):
    #     """Transform the given patch."""
    #     return patch

    # def randomize(self):
    #     """Randomize the parameters of the augmenter."""
    #     return


class ColorAugmenterBase(AugmenterBase):
    """Base class for color patch augmentation."""

    def __init__(self, keyword):
        """
        Initialize the object.
        Args:
            keyword (str): Short name for the transformation.
        """

        # Initialize the base class.
        super().__init__(keyword=keyword)


class HedColorAugmenter(ColorAugmenterBase):
    """Apply color correction in HED color space on the RGB patch."""

    def __init__(
        self,
        haematoxylin_sigma_range: Union[tuple, None],
        haematoxylin_bias_range: Union[tuple, None],
        eosin_sigma_range: Union[tuple, None],
        eosin_bias_range: Union[tuple, None],
        dab_sigma_range: Union[tuple, None],
        dab_bias_range: Union[tuple, None],
        cutoff_range: Union[tuple, None],
    ) -> ColorAugmenterBase:
        """
        The following code is derived and inspired from the following sources: https://github.com/sebastianffx/stainlib.

        Args:
            haematoxylin_sigma_range (Union[tuple, None]): Adjustment range for the Haematoxylin channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.1, 0.1).
            haematoxylin_bias_range (Union[tuple, None]): Bias range for the Haematoxylin channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.2, 0.2).
            eosin_sigma_range (Union[tuple, None]): Adjustment range for the Eosin channel from the [-1.0, 1.0] range where 0.0 means no change.
            eosin_bias_range (Union[tuple, None]): Bias range for the Eosin channel from the [-1.0, 1.0] range where 0.0 means no change.
            dab_sigma_range (Union[tuple, None]): Adjustment range for the DAB channel from the [-1.0, 1.0] range where 0.0 means no change.
            dab_bias_range (Union[tuple, None]): Bias range for the DAB channel from the [-1.0, 1.0] range where 0.0 means no change.
            cutoff_range (Union[tuple, None]): Patches with mean value outside the cutoff interval will not be augmented. Values from the [0.0, 1.0] range. The RGB channel values are from the same range.

        Returns:
            ColorAugmenterBase: The color augmenter object.
        """

        # Initialize base class.
        super().__init__(keyword="hed_color")

        # Initialize members.
        self._sigma_ranges = None  # Configured sigma ranges for H, E, and D channels.
        self._bias_ranges = None  # Configured bias ranges for H, E, and D channels.
        self._cutoff_range = None  # Cutoff interval.
        self._sigmas = None  # Randomized sigmas for H, E, and D channels.
        self._biases = None  # Randomized biases for H, E, and D channels.

        # Save configuration.
        self._setsigmaranges(
            haematoxylin_sigma_range=haematoxylin_sigma_range,
            eosin_sigma_range=eosin_sigma_range,
            dab_sigma_range=dab_sigma_range,
        )
        self._setbiasranges(
            haematoxylin_bias_range=haematoxylin_bias_range,
            eosin_bias_range=eosin_bias_range,
            dab_bias_range=dab_bias_range,
        )
        self._setcutoffrange(cutoff_range=cutoff_range)

    def _setsigmaranges(
        self,
        haematoxylin_sigma_range: Union[tuple, None],
        eosin_sigma_range: Union[tuple, None],
        dab_sigma_range: Union[tuple, None],
    ):
        """
        Set the sigma intervals.

        Args:
            haematoxylin_sigma_range (Union[tuple, None]): Adjustment range for the Haematoxylin channel.
            eosin_sigma_range (Union[tuple, None]): Adjustment range for the Eosin channel.
            dab_sigma_range (Union[tuple, None]): Adjustment range for the DAB channel.
        """

        def check_sigma_range(name, given_range):
            assert given_range is None or (
                len(given_range) == 2
                and given_range[0] < given_range[1]
                and -1.0 <= given_range[0] <= 1.0
                and -1.0 <= given_range[1] <= 1.0
            ), f"Invalid range for {name}: {given_range}"

        check_sigma_range("Haematoxylin Sigma", haematoxylin_sigma_range)
        check_sigma_range("Eosin Sigma", eosin_sigma_range)
        check_sigma_range("Dab Sigma", dab_sigma_range)

        self._sigma_ranges = [
            haematoxylin_sigma_range,
            eosin_sigma_range,
            dab_sigma_range,
        ]
        self._sigmas = [
            haematoxylin_sigma_range[0]
            if haematoxylin_sigma_range is not None
            else 0.0,
            eosin_sigma_range[0] if eosin_sigma_range is not None else 0.0,
            dab_sigma_range[0] if dab_sigma_range is not None else 0.0,
        ]

    def _setbiasranges(
        self,
        haematoxylin_bias_range: Union[tuple, None],
        eosin_bias_range: Union[tuple, None],
        dab_bias_range: Union[tuple, None],
    ):
        """
        Set the bias intervals.

        Args:
            haematoxylin_bias_range (Union[tuple, None]): Bias range for the Haematoxylin channel.
            eosin_bias_range (Union[tuple, None]): Bias range for the Eosin channel.
            dab_bias_range (Union[tuple, None]): Bias range for the DAB channel.
        """

        def check_bias_range(name, given_range):
            assert given_range is None or (
                len(given_range) != 2
                or given_range[0] < given_range[1]
                or -1.0 <= given_range[0]
                or given_range[1] <= 1.0
            ), f"Invalid range for {name}: {given_range}"

        check_bias_range("Haematoxylin Bias", haematoxylin_bias_range)
        check_bias_range("Eosin Bias", eosin_bias_range)
        check_bias_range("Dab Bias", dab_bias_range)

        self._bias_ranges = [haematoxylin_bias_range, eosin_bias_range, dab_bias_range]
        self._biases = [
            haematoxylin_bias_range[0] if haematoxylin_bias_range is not None else 0.0,
            eosin_bias_range[0] if eosin_bias_range is not None else 0.0,
            dab_bias_range[0] if dab_bias_range is not None else 0.0,
        ]

    def _setcutoffrange(self, cutoff_range: Union[tuple, None]):
        """
        Set the cutoff value. Patches with mean value outside the cutoff interval will not be augmented.

        Args:
            cutoff_range (Union[tuple, None]): Cutoff range for mean value.
        """

        def check_cutoff_range(name, given_range):
            assert given_range is None or (
                len(given_range) != 2
                or given_range[0] < given_range[1]
                or 0 <= given_range[0]
                or given_range[1] <= 1.0
            ), f"Invalid range for {name}: {given_range}"

        check_cutoff_range("Cutoff", cutoff_range)

        self._cutoff_range = cutoff_range if cutoff_range is not None else [0.0, 1.0]

    ## commented the following lines because the user is never given access to this function
    # def randomize(self):
    #     """Randomize the parameters of the augmenter."""

    #     # Randomize sigma and bias for each channel.
    #     self._sigmas = [
    #         np.random.uniform(sigma_range[0], sigma_range[1]) if sigma_range else 1.0
    #         for sigma_range in self._sigma_ranges
    #     ]
    #     self._biases = [
    #         np.random.uniform(bias_range[0], bias_range[1]) if bias_range else 0.0
    #         for bias_range in self._bias_ranges
    #     ]

    def transform(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Apply color deformation on the patch.

        Args:
            patch (torch.Tensor): The input patch to transform.

        Returns:
            torch.Tensor: The transformed patch.
        """

        current_patch = patch.numpy().astype(np.float32)
        patch_mean = (
            np.mean(current_patch) / 255.0
            if current_patch.dtype.kind != "f"
            else np.mean(current_patch)
        )

        if self._cutoff_range[0] <= patch_mean <= self._cutoff_range[1]:
            # Convert the image patch to HED color coding.
            patch_hed = rgb2hed(current_patch)

            # Augment the channels.
            for i in range(3):
                if self._sigmas[i] != 0.0:
                    patch_hed[..., i] *= 1.0 + self._sigmas[i]
                if self._biases[i] != 0.0:
                    patch_hed[..., i] += self._biases[i]

            # Convert back to RGB color coding.
            patch_transformed = hed2rgb(patch_hed)
            patch_transformed = np.clip(patch_transformed, 0.0, 1.0)

            # Convert back to integral data type if the input was also integral.
            if current_patch.dtype.kind != "f":
                patch_transformed *= 255.0
                patch_transformed = patch_transformed.astype(np.uint8)

            return patch_transformed

        # The image patch is outside the cutoff interval.
        return patch


class RandomHEDTransform(RandomTransform, IntensityTransform):
    def __init__(
        self,
        haematoxylin_sigma_range: Union[float, Tuple[float, float]] = 0.1,
        haematoxylin_bias_range: Union[float, Tuple[float, float]] = 0.1,
        eosin_sigma_range: Union[float, Tuple[float, float]] = 0.1,
        eosin_bias_range: Union[float, Tuple[float, float]] = 0.1,
        dab_sigma_range: Union[float, Tuple[float, float]] = 0.1,
        dab_bias_range: Union[float, Tuple[float, float]] = 0.1,
        cutoff_range: Union[float, Tuple[float, float]] = (0, 1),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transform_object = HedColorAugmenter(
            haematoxylin_sigma_range=haematoxylin_sigma_range,
            haematoxylin_bias_range=haematoxylin_bias_range,
            eosin_sigma_range=eosin_sigma_range,
            eosin_bias_range=eosin_bias_range,
            dab_sigma_range=dab_sigma_range,
            dab_bias_range=dab_bias_range,
            cutoff_range=cutoff_range,
        )

    def apply_transform(self, subject: Subject) -> Subject:
        # Process only if the image is RGB
        for _, image in self.get_images_dict(subject).items():
            if image.data.shape[-1] == 1:
                if image.data.ndim == 4:
                    tensor = image.data[..., 0]
                    # put channel to last axis (needed for colorconv to work)
                    tensor = tensor.permute(2, 1, 0)

                    # Apply transform
                    transformed_tensor = self.transform_object.transform(tensor)

                    # Convert tensor back to tensor data
                    transformed_data = (
                        torch.from_numpy(transformed_tensor)
                        .permute(2, 0, 1)
                        .unsqueeze(-1)
                    )

                    # Update image data
                    image.set_data(transformed_data)

        return subject
