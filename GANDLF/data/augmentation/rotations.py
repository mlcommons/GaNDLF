import torch


def tensor_rotate_90(input_image, axis):
    """
    This function rotates an image by 90 degrees around the specified axis.

    Args:
        input_image (torch.Tensor): The input tensor.
        axis (list): The axes of rotation.

    Raises:
        ValueError: If axis is not in [1, 2, 3].

    Returns:
        torch.Tensor: The rotated tensor.
    """
    # with 'axis' axis of rotation, rotate 90 degrees
    # tensor image is expected to be of shape (1, a, b, c)
    # if 0 is in axis, ensure it is not considered, since that is the batch dimension
    if 0 in axis:
        print(
            "WARNING: '0' was found in axis, adding all by '1' since '0' is batch dimension."
        )
        for count, _ in enumerate(axis):
            axis[count] += 1
    if axis not in [1, 2, 3]:
        raise ValueError("Axes must be in [1, 2, 3], but was provided as: ", axis)
    relevant_axes = set([1, 2, 3])
    affected_axes = list(relevant_axes - set([axis]))
    return torch.transpose(input_image, affected_axes[0], affected_axes[1]).flip(
        affected_axes[1]
    )


def tensor_rotate_180(input_image, axis):
    """
    This function rotates an image by 180 degrees around the specified axis.

    Args:
        input_image (torch.Tensor): The input tensor.
        axis (list): The axes of rotation.

    Raises:
        ValueError: If axis is not in [1, 2, 3].

    Returns:
        torch.Tensor: The rotated tensor.
    """
    # with 'axis' axis of rotation, rotate 180 degrees
    # tensor image is expected to be of shape (1, a, b, c)
    if axis not in [1, 2, 3]:
        raise ValueError("Axes must be in [1, 2, 3], but was provided as: ", axis)
    relevant_axes = set([1, 2, 3])
    affected_axes = list(relevant_axes - set([axis]))
    return input_image.flip(affected_axes[0]).flip(affected_axes[1])
