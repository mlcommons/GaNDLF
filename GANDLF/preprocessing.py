import numpy as np

import torch
import SimpleITK as sitk
from torchvision.transforms import Normalize

from GANDLF.utils import resample_image

from torchio.data.subject import Subject
from torchio.transforms.preprocessing.intensity.normalization_transform import (
    NormalizationTransform,
    TypeMaskingMethod,
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
def threshold_transform(min_thresh, max_thresh, p=1):
    return Lambda(
        function=partial(
            threshold_intensities, min_thresh=min_thresh, max_thresh=max_thresh
        ),
        p=p,
    )


def clip_transform(min_thresh, max_thresh, p=1):
    return Lambda(
        function=partial(
            clip_intensities, min_thresh=min_thresh, max_thresh=max_thresh
        ),
        p=p,
    )

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
}

def normalize_by_val(input_tensor, mean, std):
    """
    This function returns the tensor normalized by these particular values
    """
    normalizer = Normalize(mean, std)
    return normalizer(input_tensor)


def normalize_imagenet(input_tensor):
    """
    This function returns the tensor normalized by standard imagenet values
    """
    return normalize_by_val(
        input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )


def normalize_standardize(input_tensor):
    """
    This function returns the tensor normalized by subtracting 128 and dividing by 128
    image = (image - 128)/128
    """
    return normalize_by_val(input_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def normalize_div_by_255(input_tensor):
    """
    This function divides all values of the input tensor by 255 on all channels
    image = image/255
    """
    return normalize_by_val(input_tensor, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])


def threshold_intensities(input_tensor, min_thresh, max_thresh):
    """
    This function takes an input tensor and 2 thresholds, lower & upper and thresholds between them, basically making intensity values outside this range '0'
    """
    C = torch.zeros(input_tensor.size(), dtype=input_tensor.dtype)
    l1_tensor = torch.where(input_tensor < max_thresh, input_tensor, C)
    l2_tensor = torch.where(l1_tensor > min_thresh, l1_tensor, C)
    return l2_tensor


def clip_intensities(input_tensor, min_thresh, max_thresh):
    """
    This function takes an input tensor and 2 thresholds, lower and upper and clips between them, basically making the lowest value as 'min' and largest values as 'max'
    """
    return torch.clamp(input_tensor, min_thresh, max_thresh)


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


# adapted from https://github.com/fepegar/torchio/blob/master/torchio/transforms/preprocessing/intensity/z_normalization.py
class NonZeroNormalizeOnMaskedRegion(NormalizationTransform):
    """
    Subtract mean and divide by standard deviation.

    Args:
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional keyword arguments.
    """

    def __init__(self, masking_method: TypeMaskingMethod = None, **kwargs):
        super().__init__(masking_method=masking_method, **kwargs)
        self.args_names = ("masking_method",)

    def apply_normalization(
        self,
        subject: Subject,
        image_name: str,
        mask: torch.Tensor,
    ) -> None:
        image = subject[image_name]
        mask = image.data != 0
        standardized = self.znorm(
            image.data,
            mask,
        )
        if standardized is None:
            message = (
                "Standard deviation is 0 for masked values"
                f' in image "{image_name}" ({image.path})'
            )
            raise RuntimeError(message)
        subject.get_images_dict(intensity_only=True)[image_name]["data"] = standardized

    @staticmethod
    def znorm(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clone().float()
        values = tensor.masked_select(mask)
        mean, std = values.mean(), values.std()
        if std == 0:
            return None
        tensor[mask] -= mean
        tensor[mask] /= std
        return tensor