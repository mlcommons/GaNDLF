import torch

from torchio.data.subject import Subject
from torchio.transforms.intensity_transform import IntensityTransform

# adapted from GANDLF's NonZeroNormalizeOnMaskedRegion class
class ThresholdOrClipTransform(IntensityTransform):
    """
    Threshold or clip/clamp the intensities of the images in the subject.

        **kwargs: See :class:`~torchio.transforms.Transform` for additional keyword arguments.
    """

    def __init__(self, min_threshold=None, max_threshold=None, method=None, **kwargs):
        super().__init__(min_threshold, max_threshold, **kwargs)
        self.min_thresh = min_threshold
        self.max_thresh = max_threshold
        self.method = method
        self.probability = 1

    def apply_transform(self, subject: Subject) -> Subject:
        for image_name, _ in self.get_images_dict(subject).items():
            self.apply_threshold(subject, image_name, self.method)
        return subject

    def apply_threshold(
        self,
        subject: Subject,
        image_name: str,
        method: str,
    ) -> None:
        image = subject[image_name].data
        if method == "threshold":
            C = torch.zeros(image.shape, dtype=image.dtype)
            l1_tensor = torch.where(image < self.max_thresh, image, C)
            output = torch.where(l1_tensor > self.min_thresh, l1_tensor, C)
        elif method == "clip":
            output = torch.clamp(image, self.min_thresh, self.max_thresh)

        if output is None:
            message = (
                "Resultantant image is 0" f' in image "{image_name}" ({image.path})'
            )
            raise RuntimeError(message)
        subject.get_images_dict(intensity_only=True)[image_name]["data"] = output


# the "_transform" functions return lambdas that can be used to wrap into a Compose class
def threshold_transform(parameters):
    return ThresholdOrClipTransform(
        min_threshold=parameters["min"],
        max_threshold=parameters["max"],
        method="threshold",
    )


def clip_transform(parameters):
    return ThresholdOrClipTransform(
        min_threshold=parameters["min"], max_threshold=parameters["max"], method="clip"
    )
