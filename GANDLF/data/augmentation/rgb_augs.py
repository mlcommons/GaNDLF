from typing import Tuple, Union

import numpy as np
from torchio import Subject
from torchio.transforms import IntensityTransform
from torchio.transforms.augmentation import RandomTransform
from torchvision.transforms import ColorJitter


def colorjitter_transform(parameters):
    return RandomColorJitter(
        brightness=parameters["brightness"],
        contrast=parameters["contrast"],
        saturation=parameters["saturation"],
        hue=parameters["hue"],
    )


def hed_transform(parameters):
    return HedColorAugmenter_gandlf(
        haematoxylin_sigma_range=parameters["haematoxylin_sigma_range"],
        haematoxylin_bias_range=parameters["haematoxylin_bias_range"],
        eosin_sigma_range=parameters["eosin_sigma_range"],
        eosin_bias_range=parameters["eosin_bias_range"],
        dab_sigma_range=parameters["dab_sigma_range"],
        dab_bias_range=parameters["dab_bias_range"],
        cutoff_range=parameters["cutoff_range"],
    )


class AugmenterBase(object):
    """Base class for patch augmentation."""

    def __init__(self, keyword):
        """
        Args:
            keyword (str): Short name for the transformation.
        """
        super().__init__()
        self._keyword = keyword

    @property
    def keyword(self):
        """
        Get the keyword for the augmenter.
        Returns:
            str: Keyword.
        """

        return self._keyword

    def shapes(self, target_shapes):
        """
        Calculate the required shape of the input to achieve the target output shape.
        Args:
            target_shapes (dict): Target output shape per level.
        Returns:
            (dict): Required input shape per level.
        """

        # By default the output shapes match the input shapes.
        return target_shapes

    def transform(self, patch):
        """
        Transform the given patch.
        Args:
            patch (np.ndarray): Patch to transform.
        Returns:
            np.ndarray: Transformed patch.
        """

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
            InvalidHaematoxylinSigmaRangeError: The sigma range for Haematoxylin channel adjustment is not valid.
            InvalidEosinSigmaRangeError: The sigma range for Eosin channel adjustment is not valid.
            InvalidDabSigmaRangeError: The sigma range for DAB channel adjustment is not valid.
        """

        # Check the intervals.
        if haematoxylin_sigma_range is not None:
            if (
                len(haematoxylin_sigma_range) != 2
                or haematoxylin_sigma_range[1] < haematoxylin_sigma_range[0]
                or haematoxylin_sigma_range[0] < -1.0
                or 1.0 < haematoxylin_sigma_range[1]
            ):
                raise InvalidRangeError("Haematoxylin Sigma", haematoxylin_sigma_range)

        if eosin_sigma_range is not None:
            if (
                len(eosin_sigma_range) != 2
                or eosin_sigma_range[1] < eosin_sigma_range[0]
                or eosin_sigma_range[0] < -1.0
                or 1.0 < eosin_sigma_range[1]
            ):
                raise InvalidRangeError("Eosin Sigma", eosin_sigma_range)

        if dab_sigma_range is not None:
            if (
                len(dab_sigma_range) != 2
                or dab_sigma_range[1] < dab_sigma_range[0]
                or dab_sigma_range[0] < -1.0
                or 1.0 < dab_sigma_range[1]
            ):
                raise InvalidRangeError("Dab Sigma", dab_sigma_range)

        # Store the settings.
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
            InvalidHaematoxylinBiasRangeError: The bias range for Haematoxylin channel adjustment is not valid.
            InvalidEosinBiasRangeError: The bias range for Eosin channel adjustment is not valid.
            InvalidDabBiasRangeError: The bias range for DAB channel adjustment is not valid.
        """

        # Check the intervals.
        if haematoxylin_bias_range is not None:
            if (
                len(haematoxylin_bias_range) != 2
                or haematoxylin_bias_range[1] < haematoxylin_bias_range[0]
                or haematoxylin_bias_range[0] < -1.0
                or 1.0 < haematoxylin_bias_range[1]
            ):
                raise InvalidRangeError("Haematoxylin Bias", haematoxylin_bias_range)

        if eosin_bias_range is not None:
            if (
                len(eosin_bias_range) != 2
                or eosin_bias_range[1] < eosin_bias_range[0]
                or eosin_bias_range[0] < -1.0
                or 1.0 < eosin_bias_range[1]
            ):
                raise InvalidRangeError("Eosin Bias", eosin_bias_range)

        if dab_bias_range is not None:
            if (
                len(dab_bias_range) != 2
                or dab_bias_range[1] < dab_bias_range[0]
                or dab_bias_range[0] < -1.0
                or 1.0 < dab_bias_range[1]
            ):
                raise InvalidRangeError("Dab Bias", dab_bias_range)

        # Store the settings.
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
            cutoff_range (tuple, None): Patches with mean value outside the cutoff interval will not be augmented.
        Raises:
            InvalidCutoffRangeError: The cutoff range is not valid.
        """

        # Check the interval.
        if cutoff_range is not None:
            if (
                len(cutoff_range) != 2
                or cutoff_range[1] < cutoff_range[0]
                or cutoff_range[0] < 0.0
                or 1.0 < cutoff_range[1]
            ):
                raise InvalidRangeError("Cutoff", cutoff_range)

        # Store the setting.
        self._cutoff_range = cutoff_range if cutoff_range is not None else [0.0, 1.0]

    def transform(self, patch):
        """
        Apply color deformation on the patch.
        Args:
            patch (np.ndarray): Patch to transform.
        Returns:
            np.ndarray: Transformed patch.
        """

        # Check if the patch is inside the cutoff values.
        if patch.dtype.kind == "f":
            patch_mean = np.mean(a=patch)
        else:
            patch_mean = np.mean(a=patch.astype(dtype=np.float32)) / 255.0

        if self._cutoff_range[0] <= patch_mean <= self._cutoff_range[1]:
            # Convert the image patch to HED color coding.
            patch_hed = skimage.color.rgb2hed(rgb=patch)

            # Augment the Haematoxylin channel.
            if self._sigmas[0] != 0.0:
                patch_hed[:, :, 0] *= 1.0 + self._sigmas[0]

            if self._biases[0] != 0.0:
                patch_hed[:, :, 0] += self._biases[0]

            # Augment the Eosin channel.
            if self._sigmas[1] != 0.0:
                patch_hed[:, :, 1] *= 1.0 + self._sigmas[1]

            if self._biases[1] != 0.0:
                patch_hed[:, :, 1] += self._biases[1]

            # Augment the DAB channel.
            if self._sigmas[2] != 0.0:
                patch_hed[:, :, 2] *= 1.0 + self._sigmas[2]

            if self._biases[2] != 0.0:
                patch_hed[:, :, 2] += self._biases[2]

            # Convert back to RGB color coding.
            patch_rgb = skimage.color.hed2rgb(hed=patch_hed)
            patch_transformed = np.clip(a=patch_rgb, a_min=0.0, a_max=1.0)

            # Convert back to integral data type if the input was also integral.
            if patch.dtype.kind != "f":
                patch_transformed *= 255.0
                patch_transformed = patch_transformed.astype(dtype=np.uint8)

            return patch_transformed

        else:
            # The image patch is outside the cutoff interval.
            return patch

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize sigma and bias for each channel.
        self._sigmas = [
            np.random.uniform(low=sigma_range[0], high=sigma_range[1], size=None)
            if sigma_range is not None
            else 1.0
            for sigma_range in self._sigma_ranges
        ]
        self._biases = [
            np.random.uniform(low=bias_range[0], high=bias_range[1], size=None)
            if bias_range is not None
            else 0.0
            for bias_range in self._bias_ranges
        ]


class HedColorAugmenter_gandlf(RandomTransform, IntensityTransform):
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
        self.haematoxylin_sigma_range = haematoxylin_sigma_range
        self.haematoxylin_bias_range = haematoxylin_bias_range
        self.eosin_sigma_range = eosin_sigma_range
        self.eosin_bias_range = eosin_bias_range
        self.dab_sigma_range = dab_sigma_range
        self.dab_bias_range = dab_bias_range
        self.cutoff_range = cutoff_range

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            self.apply_to_image(image)
        # if a range is not specified, use the single value to let stainlib handle stochastic process
        if min(self.haematoxylin_sigma_range) == max(self.haematoxylin_sigma_range):
            self.haematoxylin_sigma_range = min(self.haematoxylin_sigma_range)
        else:
            self.haematoxylin_sigma_range = min(self.haematoxylin_sigma_range)
        if min(self.haematoxylin_bias_range) == max(self.haematoxylin_bias_range):
            self.haematoxylin_bias_range = min(self.haematoxylin_bias_range)
        else:
            self.haematoxylin_bias_range = min(self.haematoxylin_bias_range)
        if min(self.eosin_sigma_range) == max(self.eosin_sigma_range):
            self.eosin_sigma_range = min(self.eosin_sigma_range)
        else:
            self.eosin_sigma_range = min(self.eosin_sigma_range)
        if min(self.eosin_bias_range) == max(self.eosin_bias_range):
            self.eosin_bias_range = min(self.eosin_bias_range)
        else:
            self.eosin_bias_range = min(self.eosin_bias_range)
        if min(self.dab_sigma_range) == max(self.dab_sigma_range):
            self.dab_sigma_range = min(self.dab_sigma_range)
        else:
            self.dab_sigma_range = min(self.dab_sigma_range)
        if min(self.dab_bias_range) == max(self.dab_bias_range):
            self.dab_bias_range = min(self.dab_bias_range)
        else:
            self.dab_bias_range = min(self.dab_bias_range)
        if min(self.cutoff_range) == max(self.cutoff_range):
            self.cutoff_range = min(self.cutoff_range)
        else:
            self.cutoff_range = min(self.cutoff_range)

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


class RandomColorJitter(RandomTransform, IntensityTransform):
    r"""Add color jitter noise with random parameters.

    Add color jitter with random parameters.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
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
