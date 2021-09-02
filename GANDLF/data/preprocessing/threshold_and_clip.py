from functools import partial
import torch

from torchio.transforms import Lambda
from torchio.data.subject import Subject
from torchio.transforms.intensity_transform import IntensityTransform


def clip_intensities(input_tensor, min_thresh, max_thresh):
    """
    This function takes an input tensor and 2 thresholds, lower and upper and clips between them, basically making the lowest value as 'min' and largest values as 'max'
    """
    return torch.clamp(input_tensor, min_thresh, max_thresh)


# the "_transform" functions return lambdas that can be used to wrap into a Compose class
def threshold_transform(parameters):
    return ThresholdTransform(min_threshold=parameters["min"], max_threshold=parameters["max"])


def clip_transform(parameters):
    return Lambda(
        function=partial(
            clip_intensities, min_thresh=parameters["min"], max_thresh=parameters["max"]
        ),
        p=1,
    )

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
