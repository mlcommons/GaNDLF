import torch

from torchio.transforms.intensity_transform import IntensityTransform
from torchio.data.subject import Subject
from torchvision.transforms import Normalize
from torchio.data.image import ScalarImage


class NormalizeRGB(IntensityTransform):
    """Threshold the intensities of the RGB images in the subject into a range.

    Args:
        mean: Expended means :math:`[a, b, c]` of the output image.
        std: Expected std-deviation :math:`[a', b', c']` of the output image.

    """

    def __init__(self, mean: list = None, std: list = None, **kwargs):
        super().__init__(**kwargs)
        self.mean, self.std = mean, std
        self.args_names = "mean", "std"

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            self.apply_normalize(image)
        return subject

    def apply_normalize(self, image: ScalarImage) -> None:
        image.set_data(self.normalize(image.data))

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        normalizer = Normalize(self.mean, self.std)
        return normalizer(tensor)


# the "_transform" functions return lambdas that can be used to wrap into a Compose class
def normalize_by_val_transform(mean, std):
    return NormalizeRGB(mean=mean, std=std)


def normalize_imagenet_transform():
    return NormalizeRGB(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def normalize_standardize_transform():
    return NormalizeRGB(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def normalize_div_by_255_transform():
    return NormalizeRGB(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
