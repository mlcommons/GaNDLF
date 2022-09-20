import numpy as np
from GANDLF.utils.generic import get_array_from_image_or_tensor


def get_mapped_label(input_tensor, params):
    """
    This function maps the input label to the output label.
    Args:
        input_tensor (Union[torch.Tensor, sitk.Image]): The input label.
        params (dict): The parameters dict.

    Returns:
        np.ndarray: The output image after morphological operations.
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
