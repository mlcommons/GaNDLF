from warnings import warn
from typing import List, Union, Dict, Callable


from .wrap_torchio import (
    mri_artifact,
    motion,
    affine,
    elastic,
    swap,
    bias,
    blur,
    noise,
    noise_var,
    gamma,
    flip,
    anisotropy,
)
from .rotations import rotate_90, rotate_180
from .rgb_augs import colorjitter_transform
from .hed_augs import hed_transform

# Defining a dictionary for augmentations - key is the string and the value is the augmentation object
global_augs_dict = {
    "affine": affine,
    "elastic": elastic,
    "kspace": mri_artifact,
    "motion": motion,
    "bias": bias,
    "blur": blur,
    "noise": noise,
    "noise_var": noise_var,
    "gamma": gamma,
    "swap": swap,
    "flip": flip,
    "rotate_90": rotate_90,
    "rotate_180": rotate_180,
    "anisotropic": anisotropy,
    "colorjitter": colorjitter_transform,
    "hed_transform": hed_transform,
}


def get_augmentation_transforms(
    augmentation_params_dict: Union[Dict[str, object], List[str]]
) -> List[Callable]:
    """
    This function gets the augmentation transformations from the parameters.

    Args:
        augmentation_params_dict (dict): The dictionary containing the parameters for the augmentation.

    Returns:
        List[Callable]: The list of augmentation to be applied.
    """
    current_augmentations = []

    # Check if user specified some augmentations without extra params
    if isinstance(augmentation_params_dict, list):
        for n, augmentation_type in enumerate(augmentation_params_dict):
            if isinstance(augmentation_type, dict):
                continue
            else:
                augmentation_params_dict[n] = {augmentation_type: {}}

    for augmentation_type, augmentation_params in augmentation_params_dict.items():
        augmentation_type_lower = augmentation_type.lower()

        if augmentation_type_lower in global_augs_dict:
            current_augmentations.append(
                global_augs_dict[augmentation_type_lower](augmentation_params)
            )
        else:
            warn(
                f"Augmentation {augmentation_type} not found in the global augmentation dictionary.",
                UserWarning,
            )

    return current_augmentations
