from functools import partial
import torch

from torchio.transforms import (
    Lambda,
)


## define the functions that need to wrapped into lambdas
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


# the "_transform" functions return lambdas that can be used to wrap into a Compose class
def threshold_transform(parameters):
    return Lambda(
        function=partial(
            threshold_intensities, min_thresh=parameters["min"], max_thresh=parameters["max"]
        ),
        p=1,
    )


def clip_transform(parameters):
    return Lambda(
        function=partial(
            clip_intensities, min_thresh=parameters["min"], max_thresh=parameters["max"]
        ),
        p=1,
    )
