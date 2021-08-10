from functools import partial

import numpy as np
import torch
import SimpleITK as sitk
from torchvision.transforms import Normalize

from GANDLF.utils import resample_image
from .crop_zero_planes import CropExternalZeroplanes
from .non_zero_normalize import NonZeroNormalizeOnMaskedRegion
from .threshold_and_clip import (
    threshold_transform,
    clip_transform,
)
from .normalize_rgb import (
    normalize_by_val,
    normalize_imagenet,
    normalize_standardize,
    normalize_div_by_255,
)

from torchio.transforms import (    
    ZNormalization,
    Lambda,
)



def positive_voxel_mask(image):
    return image > 0


def nonzero_voxel_mask(image):
    return image != 0


def crop_external_zero_planes(patch_size, p=1):
    # p is only accepted as a parameter to capture when values other than one are attempted
    if p != 1:
        raise ValueError(
            "crop_external_zero_planes cannot be performed with non-1 probability."
        )
    return CropExternalZeroplanes(patch_size=patch_size)


## lambdas for pre-processing


def resize_image_resolution(input_image, output_size):
    """
    This function gets the output image spacing based on the input image and output size
    """
    inputSize = input_image.GetSize()
    outputSpacing = np.array(input_image.GetSpacing())
    for i, n in enumerate(output_size):
        outputSpacing[i] = outputSpacing[i] * (inputSize[i] / n)
    return outputSpacing


def apply_resize(input_image, preprocessing_params, interpolator=sitk.sitkLinear):
    """
    This function resizes the input image based on the output size and interpolator
    """
    return resample_image(
        input_image,
        resize_image_resolution(input_image, preprocessing_params["resize"]),
        interpolator=interpolator,
    )


def get_tensor_for_dataloader(input_sitk_image):
    """
    This function obtains the tensor to load into the data loader
    """
    temp_array = sitk.GetArrayFromImage(input_sitk_image)
    if (
        temp_array.dtype == np.uint16
    ):  # this is a contingency, because torch cannot convert this
        temp_array = temp_array.astype(np.int32)
    input_image_tensor = torch.from_numpy(temp_array).unsqueeze(
        0
    )  # single unsqueeze is always needed
    if len(input_image_tensor.shape) == 3:  # this is for 2D images
        input_image_tensor = input_image_tensor.unsqueeze(0)
    return input_image_tensor



# defining dict for pre-processing - key is the string and the value is the transform object
global_preprocessing_dict = {
    "threshold": threshold_transform,
    "clip": clip_transform,
    "normalize": ZNormalization(),
    "normalize_positive": ZNormalization(masking_method=positive_voxel_mask),
    "normalize_nonZero": ZNormalization(masking_method=nonzero_voxel_mask),
    "normalize_nonZero_masked": NonZeroNormalizeOnMaskedRegion(),
    "crop_external_zero_planes": crop_external_zero_planes,
    "normalize_imagenet": normalize_imagenet,
    "normalize_standardize": normalize_standardize,
    "normalize_div_by_255": normalize_div_by_255,
    "normalize_by_val": normalize_by_val,
}