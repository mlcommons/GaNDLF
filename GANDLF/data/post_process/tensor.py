import torch
import numpy as np


def get_mapped_label(input, params):
    """
    This function maps the input label to the output label.
    Args:
        input (torch.Tensor): The input label.
        params (dict): The parameters dict.
    Returns:
        torch.Tensor: The output image after morphological operations.
    """
    input_arr = input
    if torch.is_tensor(input):
        input_arr = input.numpy()

    if "data_postprocessing" not in params:
        return input_arr
    if "mapping" not in params["data_postprocessing"]:
        return input_arr

    mapping = params["data_postprocessing"]["mapping"]

    output = np.zeros(input_arr.shape)

    for key, value in mapping.items():
        output[input_arr == key] = value

    return output
