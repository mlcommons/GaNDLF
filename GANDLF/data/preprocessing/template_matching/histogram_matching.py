import os
import SimpleITK as sitk
import numpy as np
from torchio.data.image import ScalarImage

from .base import TemplateNormalizeBase


class HistogramMatching(TemplateNormalizeBase):
    """
    This class performs histogram matching.

    Args:
        TemplateNormalizeBase (object): Base class
    """

    def __init__(self, num_hist_level=1024, num_match_points=16, **kwargs):
        super().__init__(**kwargs)
        self.num_hist_level = num_hist_level
        self.num_match_points = num_match_points

    def apply_normalize(self, image: ScalarImage) -> None:
        image_sitk = image.as_sitk()
        if self.target is None:
            # if target is not present, perform global histogram equalization
            image_sitk_arr = sitk.GetArrayFromImage(image_sitk)
            target_arr = np.linspace(
                -1.0,
                1.0,
                np.array(image_sitk_arr.shape).prod(),
                dtype=image_sitk_arr.dtype,
            )
            target_sitk = sitk.GetImageFromArray(
                target_arr.reshape(image_sitk_arr.shape)
            )
        elif os.path.exists(self.target):
            target_sitk = sitk.ReadImage(self.target, image_sitk.GetPixelID())

        if self.target == "adaptive":
            normalized_img = sitk.AdaptiveHistogramEqualization(image_sitk)
        else:
            normalized_img = sitk.HistogramMatching(
                image_sitk, target_sitk, self.num_hist_level, self.num_match_points
            )
        image.from_sitk(normalized_img)


def histogram_matching(parameters):
    """
    This function is a wrapper for histogram matching.

    Args:
        parameters (dict): Dictionary of parameters.
    """
    num_hist_level = parameters.get("num_hist_level", 1024)
    num_match_points = parameters.get("num_match_points", 16)
    target = parameters.get("target", None)
    return HistogramMatching(
        target=target, num_hist_level=num_hist_level, num_match_points=num_match_points
    )
