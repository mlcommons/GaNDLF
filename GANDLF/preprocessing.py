import torch
import numpy as np
import SimpleITK as sitk

from torchio.data.subject import Subject
from torchio.transforms.preprocessing.intensity.normalization_transform import NormalizationTransform, TypeMaskingMethod

def threshold_intensities(input_tensor, min_val, max_val):
    '''
    This function takes an input tensor and 2 thresholds, lower & upper and thresholds between them, basically making intensity values outside this range '0'
    '''
    C = torch.zeros(input_tensor.size())
    l1_tensor = torch.where(input_tensor < max_val, input_tensor, C)
    l2_tensor = torch.where(l1_tensor > min_val, l1_tensor, C)
    return l2_tensor

def clip_intensities(input_tensor, min_val, max_val):
    '''
    This function takes an input tensor and 2 thresholds, lower and upper and clips between them, basically making the lowest value as 'min_val' and largest values as 'max_val'
    '''
    return torch.clamp(input_tensor, min_val, max_val)

def resize_image_resolution(input_image, output_size):
    '''
    This function resizes the input image based on the output size and interpolator
    '''
    inputSize = input_image.GetSize()
    outputSpacing = np.array(input_image.GetSpacing())
    for i in range(len(output_size)):
        outputSpacing[i] = outputSpacing[i] * (inputSize[i] / output_size[i])
    return outputSpacing

class NonZeroNormalize(NormalizationTransform):
    """Subtract mean and divide by standard deviation.

    Args:
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional keyword arguments.
    """
    def __init__(
            self,
            masking_method: TypeMaskingMethod = None,
            **kwargs
            ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.args_names = ('masking_method',)

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        image = subject[image_name]
        mask = image > 0
        standardized = self.znorm(
            image.data,
            mask,
        )
        if standardized is None:
            message = (
                'Standard deviation is 0 for masked values'
                f' in image "{image_name}" ({image.path})'
            )
            raise RuntimeError(message)
        image.data = standardized

    @staticmethod
    def znorm(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clone().float()
        values = tensor.masked_select(mask)
        mean, std = values.mean(), values.std()
        if std == 0:
            return None
        tensor -= mean
        tensor /= std
        return tensor

# class NonZeroNormalize:
#     '''
#     This class performs non-zero z-score normalize
#     '''
#     def __call__(self, x):
#         tensor = x.clone().float()
#         mask = tensor > 0
#         values = x > tensor.masked_select(mask)
#         mean, std = values.mean(), values.std()
#         if std == 0:
#             return None
#         tensor -= mean
#         tensor /= std
#         return tensor

class ThresholdIntensities:
    def __call__(self, x, min_val, max_val):
        input_tensor = x.clone().float()
        C = torch.zeros(input_tensor.size())
        l1_tensor = torch.where(input_tensor < max_val, input_tensor, C)
        l2_tensor = torch.where(l1_tensor > min_val, l1_tensor, C)
        return l2_tensor

class ClipIntensities:
    def __call__(self, x, min_val, max_val):
        input_tensor = x.clone().float()
        return torch.clamp(input_tensor, min_val, max_val)

class ResizeImageResolution:
    def __call__(self, x, min_val, max_val):
        input_tensor = x.clone().float()
        return torch.clamp(input_tensor, min_val, max_val)
