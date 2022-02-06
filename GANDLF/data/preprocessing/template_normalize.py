import SimpleITK as sitk

from torchio.transforms.intensity_transform import IntensityTransform
from torchio.data.subject import Subject
from torchio.data.image import ScalarImage



class TemplateNormalizeBase(IntensityTransform):
    """The Base class to apply template-based normalization techniques.

    Args:
        target (str): The target image on the basis of which the subject's images are to be normalized.
    """

    def __init__(self, target: str = None, **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.args_names = "target"

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            self.apply_normalize(image)
        return subject

    def apply_normalize(self, image: ScalarImage) -> None:
        raise NotImplementedError("This method must be implemented in a subclass.")


class HistogramMatching(TemplateNormalizeBase):
    """
    This class performs histogram matching.

    Args:
        TemplateNormalizeBase (object): Base class
    """
    
    def __init__(self, num_hist_level=1024, num_match_points=16,**kwargs):
        super().__init__(**kwargs)
        self.num_hist_level = num_hist_level
        self.num_match_points = num_match_points

    def apply_normalize(self, image: ScalarImage) -> None:
        image_sitk = image.as_sitk()
        target_sitk = sitk.ReadImage(self.target)
        return sitk.HistogramMatching(image_sitk, target_sitk, self.num_hist_level, self.num_match_points)
    