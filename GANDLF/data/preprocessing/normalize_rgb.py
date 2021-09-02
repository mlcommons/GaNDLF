import torch

from torchio.transforms.intensity_transform import IntensityTransform
from torchio.data.subject import Subject
from torchvision.transforms import Normalize

# adapted from GANDLF's ThresholdOrClipTransform class
class ThresholdRGBTransform(IntensityTransform):
    """
    Threshold the intensities of the RGB images in the subject.

        **kwargs: See :class:`~torchio.transforms.Transform` for additional keyword arguments.
    """

    def __init__(self, mean=None, std=None, p=1, **kwargs):
        super().__init__(p, mean, std, **kwargs)
        self.mean = mean
        self.std = std
        self.probability = p

    def apply_transform(self, subject: Subject) -> Subject:
        for image_name, _ in self.get_images_dict(subject).items():
            self.apply_normalize(subject, image_name)
        return subject

    def apply_normalize(
        self,
        subject: Subject,
        image_name: str,
    ) -> None:
        image = subject[image_name].data
        normalizer = Normalize(self.mean, self.std)
        output = normalizer(image)
        if output is None:
            message = (
                "Resultantant image is 0" f' in image "{image_name}" ({image.path})'
            )
            raise RuntimeError(message)
        subject.get_images_dict(intensity_only=True)[image_name]["data"] = output


# the "_transform" functions return lambdas that can be used to wrap into a Compose class
def normalize_by_val_transform(mean, std):
    return ThresholdRGBTransform(p=1, mean=mean, std=std)


def normalize_imagenet_transform():
    return ThresholdRGBTransform(p=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def normalize_standardize_transform():
    return ThresholdRGBTransform(p=1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def normalize_div_by_255_transform():
    return ThresholdRGBTransform(p=1, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
