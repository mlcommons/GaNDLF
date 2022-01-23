import torch.nn.functional as F
import numpy as np

def torch_morphological(input_image, kernel_size=1, mode='dilation'):
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
    
    if mode == 'dilation':
        output_image = max_pool(input_image, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    elif mode == 'erosion':
        output_image = -max_pool(-input_image, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    elif mode == 'closing':
        output_image = max_pool(input_image, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        output_image = -max_pool(-output_image, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    elif mode == 'opening':
        output_image = -max_pool(-input_image, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        output_image = max_pool(output_image, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    return output_image
