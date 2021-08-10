import numpy as np

from functools import partial

from torchio.transforms import (
    OneOf,
    RandomMotion,
    RandomGhosting,
    RandomSpike,
    RandomAffine,
    RandomElasticDeformation,
    RandomBiasField,
    RandomBlur,
    RandomNoise,
    RandomSwap,
    RandomAnisotropy,
    Lambda,
    RandomFlip,
    RandomGamma,
)

from .rotations import (
    tensor_rotate_90,
    tensor_rotate_180,
)

## define helper functions to create transforms
## todo: ability to change interpolation type from config file
## todo: ability to change the dimensionality according to the config file
# define individual functions/lambdas for augmentations to handle properties
def mri_artifact(p=1):
    return OneOf(
        {RandomMotion(): 0.34, RandomGhosting(): 0.33, RandomSpike(): 0.33}, p=p
    )


def affine(p=1):
    return RandomAffine(p=p)


def elastic(patch_size=None, p=1):

    if patch_size is not None:
        # define the control points and swap axes for augmentation
        num_controls = []
        for _, n in enumerate(patch_size):
            num_controls.append(max(round(n / 10), 5))  # always at least have 5
        max_displacement = np.divide(num_controls, 10)
        if num_controls[-1] == 1:
            # ensure maximum displacement is never greater than patch size
            max_displacement[-1] = 0.1
        max_displacement = max_displacement.tolist()
    else:
        # use defaults defined in torchio
        num_controls = 7
        max_displacement = 7.5
    return RandomElasticDeformation(
        num_control_points=num_controls, max_displacement=max_displacement, p=p
    )


def swap(patch_size=15, p=1):
    return RandomSwap(patch_size=patch_size, num_iterations=100, p=p)


def bias(p=1):
    return RandomBiasField(coefficients=0.5, order=3, p=p)


def blur(std, p=1):
    return RandomBlur(std=std, p=p)


def noise(mean, std, p=1):
    return RandomNoise(mean=mean, std=std, p=p)


def gamma(p=1):
    return RandomGamma(p=p)


def flip(axes=0, p=1):
    return RandomFlip(axes=axes, p=p)


def anisotropy(axes=0, downsampling=1, p=1):
    return RandomAnisotropy(
        axes=axes, downsampling=downsampling, scalars_only=True, p=p
    )


def rotate_90(axis, p=1):
    return Lambda(function=partial(tensor_rotate_90, axis=axis), p=p)


def rotate_180(axis, p=1):
    return Lambda(function=partial(tensor_rotate_180, axis=axis), p=p)


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
