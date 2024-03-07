from functools import partial
from typing import List
import torch

from torchio.transforms import Lambda


def axis_check(axis: List[int]) -> List[int]:
    """
    This function checks the axis for rotation.

    Args:
        axis (List[int]): The axes of rotation.

    Returns:
        List[int]: The affected axes.
    """
    if isinstance(axis, int):
        if axis == 0:
            axis = [1]
        else:
            axis = [axis]
    if 0 in axis:
        for count, _ in enumerate(axis):
            axis[count] += 1
    for sub_ax in axis:
        assert isinstance(
            sub_ax, int
        ), f"Axis must be an integer, but was provided as: {sub_ax}"
        assert sub_ax in [
            1,
            2,
            3,
        ], f"Axes must be in [1, 2, 3], but was provided as: {sub_ax}"

    relevant_axes = set([1, 2, 3])
    if relevant_axes == set(axis):
        affected_axes = list(relevant_axes)
    else:
        affected_axes = list(relevant_axes - set(axis))

    if len(affected_axes) == 1:
        affected_axes.append(affected_axes[0])
    return affected_axes


def tensor_rotate_90(input_image: torch.Tensor, axis: List[int]) -> torch.Tensor:
    """
    This function rotates an image by 90 degrees around the specified axis.

    Args:
        input_image (torch.Tensor): The input tensor.
        axis (List[int]): The axes of rotation.

    Returns:
        torch.Tensor: The rotated tensor.
    """
    # with 'axis' axis of rotation, rotate 90 degrees
    # tensor image is expected to be of shape (1, a, b, c)
    # if 0 is in axis, ensure it is not considered, since that is the batch dimension
    affected_axes = axis_check(axis)
    return torch.transpose(input_image, affected_axes[0], affected_axes[1]).flip(
        affected_axes[1]
    )


def tensor_rotate_180(input_image: torch.Tensor, axis: List[int]) -> torch.Tensor:
    """
    This function rotates an image by 180 degrees around the specified axis.

    Args:
        input_image (torch.Tensor): The input tensor.
        axis (List[int]): The axes of rotation.

    Returns:
        torch.Tensor: The rotated tensor.
    """
    # with 'axis' axis of rotation, rotate 180 degrees
    # tensor image is expected to be of shape (1, a, b, c)
    affected_axes = axis_check(axis)
    return input_image.flip(affected_axes[0]).flip(affected_axes[1])


def rotate_90(parameters: dict) -> Lambda:
    """
    This function rotates an image by 90 degrees around the specified axis.

    Args:
        parameters (dict): The parameters for the rotation.

    Returns:
        Lambda: The rotation function.
    """
    return Lambda(
        function=partial(tensor_rotate_90, axis=parameters["axis"]),
        p=parameters["probability"],
    )


def rotate_180(parameters: dict) -> Lambda:
    """
    This function rotates an image by 180 degrees around the specified axis.

    Args:
        parameters (dict): The parameters for the rotation.

    Returns:
        Lambda: The rotation function.
    """
    return Lambda(
        function=partial(tensor_rotate_180, axis=parameters["axis"]),
        p=parameters["probability"],
    )
