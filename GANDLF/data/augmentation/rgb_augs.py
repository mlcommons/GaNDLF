from functools import partial
from torchvision.transforms import ColorJitter
from torchio.transforms import Lambda
from collections import defaultdict
from typing import Tuple, Union, Dict, Sequence

from torchio.transforms.augmentation import RandomTransform
from torchio.transforms import IntensityTransform
from torchio import Subject


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
            brightness, "brightness", min_constraint=0
        )
        self.contrast_range = self._parse_range(contrast, "contrast", min_constraint=0)
        self.saturation_range = self._parse_range(
            saturation, "saturation", min_constraint=0
        )
        self.hue_range = self._parse_range(
            hue, "hue", min_constraint=-0.5, max_constraint=0.5
        )

    def apply_transform(self, subject: Subject) -> Subject:
        brightness = self.sample_uniform(*self.brightness_range).item()
        contrast = self.sample_uniform(*self.contrast_range).item()
        saturation = self.sample_uniform(*self.saturation_range).item()
        hue = self.sample_uniform(*self.hue_range).item()
        transform = ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
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
