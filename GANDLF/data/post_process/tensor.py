import torch


def get_mapped_label(input_tensor, params):
    """
    This function maps the input label to the output label.
    Args:
        input_tensor (torch.Tensor): The input label.
        params (dict): The parameters dict.

    Returns:
        torch.Tensor: The output image after morphological operations.
    """
    if "data_postprocessing" not in params:
        return input_tensor
    if "mapping" not in params["data_postprocessing"]:
        return input_tensor

    mapping = params["data_postprocessing"]["mapping"]

    output = torch.zeros(input_tensor.shape)

    for key, value in mapping.items():
        output[input_tensor == key] = value

    return output
