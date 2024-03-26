import numpy as np
import torch
from GANDLF.utils.generic import get_array_from_image_or_tensor


def get_mapped_label(input_tensor: torch.Tensor, params: dict) -> np.ndarray:
    """
    This function maps the input tensor to the output tensor based on the mapping provided in the params.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        params (dict): The parameters dictionary.

    Returns:
        np.ndarray: The output tensor.
    """
    input_image_array = get_array_from_image_or_tensor(input_tensor)
    if "data_postprocessing" not in params:
        return input_image_array
    if "mapping" not in params["data_postprocessing"]:
        return input_image_array

    output = np.zeros(input_image_array.shape)

    for key, value in params["data_postprocessing"]["mapping"].items():
        output[input_tensor == key] = value

    return output
