# adapted from https://github.com/fepegar/torchio/blob/main/src/torchio/transforms/augmentation/intensity/random_blur.py

from collections import defaultdict
from typing import Union, Tuple

import torch

from torchio.typing import TypeTripletFloat, TypeSextetFloat
from torchio.data.subject import Subject
from torchio.transforms import IntensityTransform
from torchio.transforms.augmentation import RandomTransform
from torchio.transforms.augmentation.intensity.random_blur import Blur


class RandomBlurEnhanced(RandomTransform, IntensityTransform):
    r"""Blur an image using a random-sized Gaussian filter.

    Args:
        std: Tuple :math:`(a_1, b_1, a_2, b_2, a_3, b_3)` representing the
            ranges (in mm) of the standard deviations
            :math:`(\sigma_1, \sigma_2, \sigma_3)` of the Gaussian kernels used
            to blur the image along each axis, where
            :math:`\sigma_i \sim \mathcal{U}(a_i, b_i)`.
            If two values :math:`(a, b)` are provided,
            then :math:`\sigma_i \sim \mathcal{U}(a, b)`.
            If only one value :math:`x` is provided,
            then :math:`\sigma_i \sim \mathcal{U}(0, x)`.
            If three values :math:`(x_1, x_2, x_3)` are provided,
            then :math:`\sigma_i \sim \mathcal{U}(0, x_i)`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, std: Union[float, Tuple[float, float]] = None, **kwargs):
        super().__init__(**kwargs)
        self.std_original = std

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        for name, image in self.get_images_dict(subject).items():
            self.std_ranges = self.calculate_std_ranges(image)
            arguments["std"][name] = self.get_params(self.std_ranges)
        transform = Blur(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(self, std_ranges: TypeSextetFloat) -> TypeTripletFloat:
        std = self.sample_uniform_sextet(std_ranges)
        return std

    def calculate_std_ranges(self, image: torch.Tensor) -> Tuple[float, float]:
        std_ranges = self.std_original
        if self.std_original is None:
            # calculate the default std range based on 1.5% of the input image std - https://github.com/mlcommons/GaNDLF/issues/518
            std_ranges = (0, 0.015 * torch.std(image.data.float()).item())
        return self.parse_params(std_ranges, None, "std", min_constraint=0)
