import torch
from typing import List, Optional

from torchio.transforms.intensity_transform import IntensityTransform
from torchio.data.subject import Subject
from torchio.data.image import ScalarImage


class NormalizeRGB(IntensityTransform):
    """Threshold the intensities of the RGB images in the subject into a range.

    Args:
        mean: Expended means :math:`[a, b, c]` of the output image.
        std: Expected std-deviation :math:`[a', b', c']` of the output image.

    """

    def __init__(
        self,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mean, self.std = mean, std
        self.args_names = "mean", "std"

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            self.apply_normalize(image)
        return subject

    def apply_normalize(self, image: ScalarImage) -> None:
        image_data = image.data
        if image_data.shape[-1] == 1:
            image_data = image_data.squeeze(-1)
            image_data = self.normalize(image_data, self.mean, self.std)
            image_data = image_data.unsqueeze(-1)
            image.set_data(image_data)
        else:
            image.set_data(self.normalize(image_data, self.mean, self.std))

    def normalize(self, tensor: torch.Tensor, mean: List[float], std: List[float]):
        """Normalize a float tensor image with mean and standard deviation.
        This transform does not support PIL image.

        Args:
            tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.

        Returns:
            Tensor: Normalized Tensor image.
        """

        # standard operation defined in ToTensor
        tensor = tensor.div(255)
        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)

        if (std == 0).any():
            raise ValueError(
                f"std evaluated to zero after conversion to {dtype}​​​​​, leading to division by zero."
            )
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        tensor = tensor.sub(mean)
        tensor = tensor.div(std)

        return tensor


# the "_transform" functions return lambdas that can be used to wrap into a Compose class
def normalize_by_val_transform(mean, std):
    return NormalizeRGB(mean=mean, std=std)


def normalize_imagenet_transform():
    return NormalizeRGB(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def normalize_standardize_transform():
    return NormalizeRGB(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def normalize_div_by_255_transform():
    return NormalizeRGB(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
