import torch
import numpy as np
from GANDLF.utils.generic import get_array_from_image_or_tensor


def get_mapped_label(input_tensor, params):
    """
    This function maps the input label to the output label.
    Args:
        input_tensor (torch.Tensor): The input label.
        params (dict): The parameters dict.

    Returns:
        torch.Tensor: The output image after morphological operations.
    """
    input_arr = get_array_from_image_or_tensor(input_tensor)

    if "data_postprocessing" not in params:
        return torch.from_numpy(input_arr)
    if "mapping" not in params["data_postprocessing"]:
        return torch.from_numpy(input_arr)

    mapping = params["data_postprocessing"]["mapping"]

    output = np.zeros(input_arr.shape)

    for key, value in mapping.items():
        output[input_arr == key] = value

    return torch.from_numpy(output)
