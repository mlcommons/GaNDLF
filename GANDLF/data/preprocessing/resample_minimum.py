from typing import Optional
import numpy as np
import SimpleITK as sitk

from torchio.transforms import Resample
from torchio.typing import TypeTripletFloat


class Resample_Minimum(Resample):
    """
    This performs resampling of an image to the minimum spacing specified by a single number. Otherwise, it will perform standard resampling.

    Args:
        Resample (SpatialTransform): The parent class for resampling.
    """

    def __init__(self, target: Optional[float] = 1, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_reference_image(
        floating_sitk: sitk.Image, spacing: TypeTripletFloat
    ) -> sitk.Image:
        old_spacing = np.array(floating_sitk.GetSpacing())
        new_spacing = np.array(spacing)
        new_spacing = np.minimum(old_spacing, new_spacing)
        old_size = np.array(floating_sitk.GetSize())
        new_size = old_size * old_spacing / new_spacing
        new_size = np.ceil(new_size).astype(np.uint16)
        new_size[old_size == 1] = 1  # keep singleton dimensions
        new_origin_index = 0.5 * (new_spacing / old_spacing - 1)
        new_origin_lps = floating_sitk.TransformContinuousIndexToPhysicalPoint(
            new_origin_index
        )
        reference = sitk.Image(
            new_size.tolist(),
            floating_sitk.GetPixelID(),
            floating_sitk.GetNumberOfComponentsPerPixel(),
        )
        reference.SetDirection(floating_sitk.GetDirection())
        reference.SetSpacing(new_spacing.tolist())
        reference.SetOrigin(new_origin_lps)
        return reference
