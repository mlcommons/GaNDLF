# adapted from https://github.com/fepegar/torchio/blob/main/src/torchio/transforms/augmentation/intensity/random_noise.py

from collections import defaultdict
from typing import Tuple, Union

import torch
from torchio.data.subject import Subject
from torchio.transforms import IntensityTransform
from torchio.transforms.augmentation import RandomTransform
from torchio.transforms.augmentation.intensity.random_noise import Noise


class RandomNoiseEnhanced(RandomTransform, IntensityTransform):
    r"""Add Gaussian noise with random parameters.

    Add noise sampled from a normal distribution with random parameters.

    Args:
        mean: Mean :math:`\mu` of the Gaussian distribution
            from which the noise is sampled.
            If two values :math:`(a, b)` are provided,
            then :math:`\mu \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`\mu \sim \mathcal{U}(-d, d)`.
        std: Standard deviation :math:`\sigma` of the Gaussian distribution
            from which the noise is sampled.
            If two values :math:`(a, b)` are provided,
            then :math:`\sigma \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`\sigma \sim \mathcal{U}(0, d)`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(
        self,
        mean: Union[float, Tuple[float, float]] = 0,
        std: Union[float, Tuple[float, float]] = (0, 0.25),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mean_range = self._parse_range(mean, "mean")
        self.std_original = std

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        for name, image in self.get_images_dict(subject).items():
            self.std_range = self.calculate_std_ranges(image)
            mean, std, seed = self.get_params(self.mean_range, self.std_range)
            arguments["mean"][name] = mean
            arguments["std"][name] = std
            arguments["seed"][name] = seed
        transform = Noise(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(
        self,
        mean_range: Tuple[float, float],
        std_range: Tuple[float, float],
    ) -> Tuple[float, float]:
        mean = self.sample_uniform(*mean_range).item()
        std = self.sample_uniform(*std_range).item()
        seed = self._get_random_seed()
        return mean, std, seed

    def calculate_std_ranges(self, image: torch.Tensor) -> tuple:
        std_ranges = self.std_original
        if self.std_original is None:
            # calculate the default std range based on 1.5% of the input image std - https://github.com/mlcommons/GaNDLF/issues/518
            std_ranges = (0, 0.015 * torch.std(image.data).item())
        return self._parse_range(std_ranges, "std", min_constraint=0)
