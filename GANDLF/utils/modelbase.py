import torch
import torch.nn.functional as F


def get_modelbase_final_layer(final_convolution_layer):
    """
    This function gets the final layer of the model.

    Args:
        final_convolution_layer (str): The final layer of the model as a string.

    Returns:
        Functional: sigmoid, softmax, or None
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
