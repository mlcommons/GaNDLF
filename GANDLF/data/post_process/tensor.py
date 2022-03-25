import torch
import numpy as np


def get_mapped_label(input_tensor, params):
    """
    This function maps the input label to the output label.
    Args:
        input_tensor (torch.Tensor): The input label.
        params (dict): The parameters dict.

    Returns:
        torch.Tensor: The output image after morphological operations.
    """
    input_arr = input_tensor
    if torch.is_tensor(input_tensor):
        input_arr = input_tensor.numpy()

    if "data_postprocessing" not in params:
        return input_arr
    if "mapping" not in params["data_postprocessing"]:
        return input_arr

    mapping = params["data_postprocessing"]["mapping"]

    output = np.zeros(input_arr.shape)

    for key, value in mapping.items():
        output[input_arr == key] = value

    return output
