from .rgb_augs import colorjitter_transform
from .rotations import rotate_90, rotate_180
from .wrap_torchio import (
    affine,
    anisotropy,
    bias,
    blur,
    elastic,
    flip,
    gamma,
    motion,
    mri_artifact,
    noise,
    swap,
)

# Defining a dictionary for augmentations - key is the string and the value is the augmentation object
global_augs_dict = {
    "affine": affine,
    "elastic": elastic,
    "kspace": mri_artifact,
    "motion": motion,
    "bias": bias,
    "blur": blur,
    "noise": noise,
    "gamma": gamma,
    "swap": swap,
    "flip": flip,
    "rotate_90": rotate_90,
    "rotate_180": rotate_180,
    "anisotropic": anisotropy,
    "colorjitter": colorjitter_transform,
}
