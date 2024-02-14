from typing import Union
import torch
import torch.nn.functional as F


def get_modelbase_final_layer(final_convolution_layer: str) -> Union[object, None]:
    """
    This function returns the final convolution layer based on the input string.

    Args:
        final_convolution_layer (str): The string representing the final convolution layer.

    Returns:
        Union[object, None]: The final convolution layer.
    """
    none_list = [
        "none",
        None,
        "None",
        "regression",
        "classification_but_not_softmax",
        "logits",
        "classification_without_softmax",
    ]

    if final_convolution_layer in ["sigmoid", "sig"]:
        final_convolution_layer = torch.sigmoid

    elif final_convolution_layer in ["softmax", "soft"]:
        final_convolution_layer = F.softmax

    elif final_convolution_layer in none_list:
        final_convolution_layer = None

    return final_convolution_layer
