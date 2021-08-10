from .wrap_torchio import (
    mri_artifact,
    affine,
    elastic,
    swap,
    bias,
    blur,
    noise,
    gamma,
    flip,
    anisotropy,
    rotate_90,
    rotate_180,
)

# Defining a dictionary for augmentations - key is the string and the value is the augmentation object
global_augs_dict = {
    "affine": affine,
    "elastic": elastic,
    "kspace": mri_artifact,
    "bias": bias,
    "blur": blur,
    "noise": noise,
    "gamma": gamma,
    "swap": swap,
    "flip": flip,
    "rotate_90": rotate_90,
    "rotate_180": rotate_180,
    "anisotropic": anisotropy,
}
