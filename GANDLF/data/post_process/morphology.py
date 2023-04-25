import torch
import torch.nn.functional as F
from skimage.measure import label
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_closing
from GANDLF.utils.generic import get_array_from_image_or_tensor


def torch_morphological(input_image, kernel_size=1, mode="dilation"):
    """
    This function enables morphological operations using torch. Adapted from https://github.com/DIVA-DIA/Generating-Synthetic-Handwritten-Historical-Documents/blob/e6a798dc2b374f338804222747c56cb44869af5b/HTR_ctc/utils/auxilary_functions.py#L10.

    Args:
        input_image (torch.Tensor): The input image.
        kernel_size (list): The size of the window to take a max over.
        mode (str): The type of morphological operation to perform.

    Returns:
        torch.Tensor: The output image after morphological operations.
    """

    if len(input_image.shape) == 4:
        max_pool = F.max_pool2d
    elif len(input_image.shape) == 5:
        max_pool = F.max_pool3d
    else:
        raise ValueError("Input image has invalid shape for morphological operations.")

    if mode == "dilation":
        output_image = max_pool(
            input_image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
    elif mode == "erosion":
        output_image = -max_pool(
            -input_image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
    elif mode == "closing":
        output_image = max_pool(
            input_image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        output_image = -max_pool(
            -output_image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
    elif mode == "opening":
        output_image = -max_pool(
            -input_image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        output_image = max_pool(
            output_image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    return output_image


def fill_holes(input_image, params=None):
    """
    This function fills holes in masks.

    Args:
        input_image (torch.Tensor): The input image.
        params (dict): The parameters dict; unused.

    Returns:
        torch.Tensor: The output image after morphological operations.
    """
    input_image_array = get_array_from_image_or_tensor(input_image).astype(int)
    input_image_array_closed = binary_closing(input_image_array)
    # Fill the holes in binary objects
    output_array = binary_fill_holes(input_image_array_closed).astype(int)

    return torch.from_numpy(output_array)


def cca(input_image, params=None):
    """
    This function performs connected component analysis on the input image.

    Args:
        input_image (torch.Tensor): The input image.
        params (dict): The parameters dict;

    Returns:
        torch.Tensor: The output image after morphological operations.
    """
    seg = get_array_from_image_or_tensor(input_image)
    mask = seg != 0

    connectivity = input_image.ndim - 1
    labels_connected = label(mask, connectivity=connectivity)
    labels_connected_sizes = [
        np.sum(labels_connected == i) for i in np.unique(labels_connected)
    ]
    largest_region = 0
    if len(labels_connected_sizes) > 1:
        largest_region = np.argmax(labels_connected_sizes[1:]) + 1
    seg[labels_connected != largest_region] = 0

    return seg
