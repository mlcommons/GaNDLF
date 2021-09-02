from functools import partial
import torch

from torchio.transforms import Lambda
from torchio.data.subject import Subject
from torchio.transforms.intensity_transform import IntensityTransform

# adapted from GANDLF's NonZeroNormalizeOnMaskedRegion class
class ThresholdTransform(IntensityTransform):
    """
    Threshold the intensities of the images in the subject.

        **kwargs: See :class:`~torchio.transforms.Transform` for additional keyword arguments.
    """

    def __init__(self, min_threshold = None, max_threshold = None, **kwargs):
        super().__init__(min_threshold, max_threshold, **kwargs)
        self.min_thresh = min_threshold
        self.max_thresh = max_threshold
        self.probability = 1

    def apply_transform(self, subject: Subject) -> Subject:
        for image_name, _ in self.get_images_dict(subject).items():
            self.apply_threshold(subject, image_name)
        return subject

    def apply_threshold(
        self,
        subject: Subject,
        image_name: str,
    ) -> None:
        image = subject[image_name].data
        C = torch.zeros(image.shape, dtype=image.dtype)
        l1_tensor = torch.where(image < self.max_thresh, image, C)
        l2_tensor = torch.where(l1_tensor > self.min_thresh, l1_tensor, C)
        
        if l2_tensor is None:
            message = (
                "Resultantant image is 0"
                f' in image "{image_name}" ({image.path})'
            )
            raise RuntimeError(message)
        subject.get_images_dict(intensity_only=True)[image_name]["data"] = l2_tensor


# adapted from GANDLF's NonZeroNormalizeOnMaskedRegion class
class ClipTransform(IntensityTransform):
    """
    Clip or clamp the intensities of the images in the subject.

        **kwargs: See :class:`~torchio.transforms.Transform` for additional keyword arguments.
    """

    def __init__(self, min_threshold = None, max_threshold = None, **kwargs):
        super().__init__(min_threshold, max_threshold, **kwargs)
        self.min_thresh = min_threshold
        self.max_thresh = max_threshold
        self.probability = 1

    def apply_transform(self, subject: Subject) -> Subject:
        for image_name, _ in self.get_images_dict(subject).items():
            self.apply_threshold(subject, image_name)
        return subject

    def apply_threshold(
        self,
        subject: Subject,
        image_name: str,
    ) -> None:
        image = subject[image_name].data
        output = torch.clamp(image, self.min_thresh, self.max_thresh)
        
        if output is None:
            message = (
                "Resultantant image is 0"
                f' in image "{image_name}" ({image.path})'
            )
            raise RuntimeError(message)
        subject.get_images_dict(intensity_only=True)[image_name]["data"] = output



# the "_transform" functions return lambdas that can be used to wrap into a Compose class
def threshold_transform(parameters):
    return ThresholdTransform(min_threshold=parameters["min"], max_threshold=parameters["max"])


def clip_transform(parameters):
    return ClipTransform(min_threshold=parameters["min"], max_threshold=parameters["max"])