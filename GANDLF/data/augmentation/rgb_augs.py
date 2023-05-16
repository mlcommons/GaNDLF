from torchvision.transforms import ColorJitter
from typing import Tuple, Union
import numpy as np
from skimage.color import rgb2hed, hed2rgb
from torchio.transforms.augmentation import RandomTransform
from torchio.transforms import IntensityTransform
from torchio import Subject
from GANDLF.utils.exceptions import InvalidRangeError


def colorjitter_transform(parameters):
    return RandomColorJitter(
        brightness=parameters["brightness"],
        contrast=parameters["contrast"],
        saturation=parameters["saturation"],
        hue=parameters["hue"],
    )


class RandomColorJitter(RandomTransform, IntensityTransform):
    r"""Add color jitter noise with random parameters.

    Add color jitter with random parameters.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non ne~gative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0.1,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.brightness_range = self._parse_range(
            brightness, "brightness", min_constraint=0, max_constraint=1
        )
        self.contrast_range = self._parse_range(
            contrast, "contrast", min_constraint=0, max_constraint=1
        )
        self.saturation_range = self._parse_range(
            saturation, "saturation", min_constraint=0, max_constraint=1
        )
        self.hue_range = self._parse_range(
            hue, "hue", min_constraint=-0.5, max_constraint=0.5
        )

    def apply_transform(self, subject: Subject) -> Subject:
        # if a range is not specified, use the single value to let torch handle stochastic process
        if min(self.brightness_range) == max(self.brightness_range):
            brightness = min(self.brightness_range)
        else:
            brightness = self.brightness_range
        if min(self.contrast_range) == max(self.contrast_range):
            contrast = min(self.contrast_range)
        else:
            contrast = self.contrast_range
        if min(self.saturation_range) == max(self.saturation_range):
            saturation = min(self.saturation_range)
        else:
            saturation = self.saturation_range
        if min(self.hue_range) == max(self.hue_range):
            hue = min(self.hue_range)
        else:
            hue = self.hue_range
        transform = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        for _, image in self.get_images_dict(subject).items():
            # proceed with processing only if the image is RGB
            if image.data.shape[-1] == 1:
                temp = image.data
                # remove last dimension
                temp = temp.squeeze(-1)
                # add a shell batch dimension
                temp = temp.unsqueeze(0)
                # apply transform
                temp = transform(temp)
                # remove shell batch dimension
                temp = temp.squeeze(0)
                # add last dimension to bring image back to original shape
                temp = temp.unsqueeze(-1)
                image.set_data(temp)

        return subject


def hed_transform(parameters):
    return HedColorAugmenter(
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

    @property
    def keyword(self):
        """Get the keyword for the augmenter."""
        return self._keyword

    def shapes(self, target_shapes):
        """Calculate the required shape of the input to achieve the target output shape."""
        return target_shapes

    def transform(self, patch):
        """Transform the given patch."""
        pass

    def randomize(self):
        """Randomize the parameters of the augmenter."""
        pass


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
        haematoxylin_sigma_range,
        haematoxylin_bias_range,
        eosin_sigma_range,
        eosin_bias_range,
        dab_sigma_range,
        dab_bias_range,
        cutoff_range,
    ):
        """
        The following code is derived and inspired from the following sources:
        https://github.com/sebastianffx/stainlib
        and it is covered under MIT license.
        Initialize the object. For each channel the augmented value is calculated as value = value * sigma + bias
        Args:
            haematoxylin_sigma_range (tuple, None): Adjustment range for the Haematoxylin channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.1, 0.1).
            haematoxylin_bias_range (tuple, None): Bias range for the Haematoxylin channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.2, 0.2).
            eosin_sigma_range (tuple, None): Adjustment range for the Eosin channel from the [-1.0, 1.0] range where 0.0 means no change.
            eosin_bias_range (tuple, None) Bias range for the Eosin channel from the [-1.0, 1.0] range where 0.0 means no change.
            dab_sigma_range (tuple, None): Adjustment range for the DAB channel from the [-1.0, 1.0] range where 0.0 means no change.
            dab_bias_range (tuple, None): Bias range for the DAB channel from the [-1.0, 1.0] range where 0.0 means no change.
            cutoff_range (tuple, None): Patches with mean value outside the cutoff interval will not be augmented. Values from the [0.0, 1.0] range. The RGB channel values are from the same range.
        Raises:
            InvalidHaematoxylinSigmaRangeError: The sigma range for Haematoxylin channel adjustment is not valid.
            InvalidHaematoxylinBiasRangeError: The bias range for Haematoxylin channel adjustment is not valid.
            InvalidEosinSigmaRangeError: The sigma range for Eosin channel adjustment is not valid.
            InvalidEosinBiasRangeError: The bias range for Eosin channel adjustment is not valid.
            InvalidDabSigmaRangeError: The sigma range for DAB channel adjustment is not valid.
            InvalidDabBiasRangeError: The bias range for DAB channel adjustment is not valid.
            InvalidCutoffRangeError: The cutoff range is not valid.
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
        self, haematoxylin_sigma_range, eosin_sigma_range, dab_sigma_range
    ):
        """
        Set the sigma intervals.
        Args:
            haematoxylin_sigma_range (tuple, None): Adjustment range for the Haematoxylin channel.
            eosin_sigma_range (tuple, None): Adjustment range for the Eosin channel.
            dab_sigma_range (tuple, None): Adjustment range for the DAB channel.
        Raises:
            InvalidRangeError: If the sigma range for any channel adjustment is not valid.
        """

        def check_sigma_range(name, range):
            if range is not None:
                if (
                    len(range) != 2
                    or range[1] < range[0]
                    or range[0] < -1.0
                    or 1.0 < range[1]
                ):
                    raise InvalidRangeError(name, range)

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

    def _setbiasranges(self, haematoxylin_bias_range, eosin_bias_range, dab_bias_range):
        """
        Set the bias intervals.
        Args:
            haematoxylin_bias_range (tuple, None): Bias range for the Haematoxylin channel.
            eosin_bias_range (tuple, None) Bias range for the Eosin channel.
            dab_bias_range (tuple, None): Bias range for the DAB channel.
        Raises:
            InvalidRangeError: If the bias range for any channel adjustment is not valid.
        """

        def check_bias_range(name, range):
            if range is not None:
                if (
                    len(range) != 2
                    or range[1] < range[0]
                    or range[0] < -1.0
                    or 1.0 < range[1]
                ):
                    raise InvalidRangeError(name, range)

        check_bias_range("Haematoxylin Bias", haematoxylin_bias_range)
        check_bias_range("Eosin Bias", eosin_bias_range)
        check_bias_range("Dab Bias", dab_bias_range)

        self._bias_ranges = [haematoxylin_bias_range, eosin_bias_range, dab_bias_range]
        self._biases = [
            haematoxylin_bias_range[0] if haematoxylin_bias_range is not None else 0.0,
            eosin_bias_range[0] if eosin_bias_range is not None else 0.0,
            dab_bias_range[0] if dab_bias_range is not None else 0.0,
        ]

    def _setcutoffrange(self, cutoff_range):
        """
        Set the cutoff value. Patches with mean value outside the cutoff interval will not be augmented.
        Args:
            cutoff_range (tuple, None): Cutoff range for mean value.
        Raises:
            InvalidRangeError: If the cutoff range is not valid.
        """

        def check_cutoff_range(name, range):
            if range is not None:
                if (
                    len(range) != 2
                    or range[1] < range[0]
                    or range[0] < 0.0
                    or 1.0 < range[1]
                ):
                    raise InvalidRangeError(name, range)

        check_cutoff_range("Cutoff", cutoff_range)

        self._cutoff_range = cutoff_range if cutoff_range is not None else [0.0, 1.0]

    def transform(self, patch):
        """
        Apply color deformation on the patch.
        Args:
            patch (np.ndarray): Patch to transform.
        Returns:
            np.ndarray: Transformed patch.
        """

        if patch.dtype.kind == "f":
            patch_mean = np.mean(patch)
        else:
            patch_mean = np.mean(patch.astype(np.float32)) / 255.0

        if self._cutoff_range[0] <= patch_mean <= self._cutoff_range[1]:
            # Convert the image patch to HED color coding.
            patch_hed = rgb2hed(patch)

            # Augment the channels.
            for i in range(3):
                if self._sigmas[i] != 0.0:
                    patch_hed[:, :, i] *= 1.0 + self._sigmas[i]
                if self._biases[i] != 0.0:
                    patch_hed[:, :, i] += self._biases[i]

            # Convert back to RGB color coding.
            patch_rgb = hed2rgb(patch_hed)
            patch_transformed = np.clip(patch_rgb, 0.0, 1.0)

            # Convert back to integral data type if the input was also integral.
            if patch.dtype.kind != "f":
                patch_transformed *= 255.0
                patch_transformed = patch_transformed.astype(np.uint8)

            return patch_transformed
        else:
            # The image patch is outside the cutoff interval.
            return patch

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize sigma and bias for each channel.
        self._sigmas = [
            np.random.uniform(sigma_range[0], sigma_range[1]) if sigma_range else 1.0
            for sigma_range in self._sigma_ranges
        ]
        self._biases = [
            np.random.uniform(bias_range[0], bias_range[1]) if bias_range else 0.0
            for bias_range in self._bias_ranges
        ]


class HedColorAugmenter(RandomTransform, IntensityTransform):
    r"""H&E color augmentation.
    Args:
        haematoxylin_sigma_range: Range of sigma for haematoxylin channel.
        haematoxylin_bias_range: Range of bias for haematoxylin channel.
        eosin_sigma_range: Range of sigma for eosin channel.
        eosin_bias_range: Range of bias
        dab_sigma_range: Range of sigma for DAB channel.
        dab_bias_range: Range of bias for DAB channel.
        cutoff_range: Range of cutoff for DAB channel.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.
    """

    def __init__(
        self,
        haematoxylin_sigma_range: Tuple[float, float] = (0.5, 1.5),
        haematoxylin_bias_range: Tuple[float, float] = (-0.5, 0.5),
        eosin_sigma_range: Tuple[float, float] = (0.5, 1.5),
        eosin_bias_range: Tuple[float, float] = (-0.5, 0.5),
        dab_sigma_range: Tuple[float, float] = (0.5, 1.5),
        dab_bias_range: Tuple[float, float] = (-0.5, 0.5),
        cutoff_range: Tuple[float, float] = (0.5, 1.5),
        p: float = 1,
        seed: Union[int, None] = None,
    ):
        super().__init__(p=p, seed=seed)
        self.haematoxylin_sigma_range = min(haematoxylin_sigma_range)
        self.haematoxylin_bias_range = min(haematoxylin_bias_range)
        self.eosin_sigma_range = min(eosin_sigma_range)
        self.eosin_bias_range = min(eosin_bias_range)
        self.dab_sigma_range = min(dab_sigma_range)
        self.dab_bias_range = min(dab_bias_range)
        self.cutoff_range = min(cutoff_range)

    def apply_transform(self, subject: Subject) -> Subject:
        transform = HedColorAugmenter(
            haematoxylin_sigma_range=self.haematoxylin_sigma_range,
            haematoxylin_bias_range=self.haematoxylin_bias_range,
            eosin_sigma_range=self.eosin_sigma_range,
            eosin_bias_range=self.eosin_bias_range,
            dab_sigma_range=self.dab_sigma_range,
            dab_bias_range=self.dab_bias_range,
            cutoff_range=self.cutoff_range,
        )
        for _, image in self.get_images_dict(subject).items():
            if image.data.shape[-1] == 1:
                temp = image.data.squeeze(-1).unsqueeze(0)
                temp = transform(temp).squeeze(0).unsqueeze(-1)
                image.set_data(temp)
        return subject
