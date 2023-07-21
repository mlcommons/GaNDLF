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
from .rotations import (
    rotate_90,
    rotate_180,
)
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
