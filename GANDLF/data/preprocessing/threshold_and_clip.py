from typing import Optional
import torch

from torchio.data.subject import Subject
from torchio.data.image import ScalarImage
from torchio.transforms.intensity_transform import IntensityTransform
from torchio.transforms.preprocessing.intensity.clamp import Clamp


class Threshold(IntensityTransform):
    """Threshold intensity values into a range :math:`[a, b]`.

    Args:
        out_min: Minimum value :math:`a` of the output image. If ``None``, the
            minimum of the image is used.
        out_max: Maximum value :math:`b` of the output image. If ``None``, the
            maximum of the image is used.

    Example:
        >>> import torchio as tio
        >>> ct = tio.datasets.Slicer('CTChest').CT_chest
        >>> HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 1000
        >>> threshold = tio.Threshold(out_min=HOUNSFIELD_AIR, out_max=HOUNSFIELD_BONE)
        >>> ct_thresholded = threshold(ct)

    .. plot::

        import torchio as tio
        subject = tio.datasets.Slicer('CTChest')
        ct = subject.CT_chest
        HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 1000
        threshold = tio.Threshold(out_min=HOUNSFIELD_AIR, out_max=HOUNSFIELD_BONE)
        ct_thresholded = threshold(ct)
        subject.add_image(ct_thresholded, 'Thresholded')
        subject.plot()

    """

    def __init__(
        self, out_min: Optional[float] = None, out_max: Optional[float] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.out_min, self.out_max = out_min, out_max
        self.args_names = "out_min", "out_max"

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            self.apply_threshold(image)
        return subject

    def apply_threshold(self, image: ScalarImage) -> None:
        image.set_data(self.threshold(image.data))

    def threshold(self, tensor: torch.Tensor) -> torch.Tensor:
        C = torch.zeros(tensor.shape, dtype=tensor.dtype)
        l1_tensor = torch.where(tensor < self.out_max, tensor, C)
        output = torch.where(l1_tensor > self.out_min, l1_tensor, C)
        return output


# the "_transform" functions return lambdas that can be used to wrap into a Compose class
def threshold_transform(parameters: dict) -> Threshold:
    """
    This function returns a lambda function that can be used to wrap into a Compose class.

    Args:
        parameters (dict): The parameters dictionary.

    Returns:
        Threshold: The transform to threshold the image.
    """
    return Threshold(out_min=parameters["min"], out_max=parameters["max"])


def clip_transform(parameters: dict) -> Clamp:
    """
    This function returns a lambda function that can be used to wrap into a Compose class.

    Args:
        parameters (dict): The parameters dictionary.

    Returns:
        Clamp: The transform to clip the image.
    """
    return Clamp(out_min=parameters["min"], out_max=parameters["max"])
