import torch


def get_mapped_label(input, params):
    """
    This function maps the input label to the output label.
    Args:
        input (torch.Tensor): The input label.
        params (dict): The parameters dict.
    Returns:
        torch.Tensor: The output image after morphological operations.
    """
    if "post_processing" not in params:
        return input
    if "mapping" not in params["post_processing"]:
        return input

    mapping = params["post_processing"]["mapping"]

    output = torch.zeros(input.shape)

    for key, value in mapping.items():
        output[input == key] = value

    return output
