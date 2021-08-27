import numpy as np

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
    RandomFlip,
    RandomGamma,
)

## define helper functions to create transforms
## todo: ability to change interpolation type from config file
## todo: ability to change the dimensionality according to the config file
# define individual functions/lambdas for augmentations to handle properties
def mri_artifact(parameters):
    return OneOf(
        {RandomMotion(): 0.34, RandomGhosting(): 0.33, RandomSpike(): 0.33}, p=parameters["probability"]
    )


def affine(parameters):
    return RandomAffine(p=parameters["probability"])


def elastic(parameters):

    if parameters["patch_size"] is not None:
        # define the control points and swap axes for augmentation
        num_controls = []
        for _, n in enumerate(parameters["patch_size"]):
            num_controls.append(max(n, 5))  # always at least have 5
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
        num_control_points=num_controls, max_displacement=max_displacement, p=parameters["probability"]
    )


def swap(parameters):
    return RandomSwap(patch_size=parameters["patch_size"], num_iterations=100, p=parameters["probability"])


def bias(parameters):
    return RandomBiasField(coefficients=0.5, order=3, p=parameters["probability"])


def blur(parameters, std=1):
    return RandomBlur(std=std, p=parameters["probability"])


def noise(parameters, mean=0, std=1):
    return RandomNoise(mean=mean, std=std, p=parameters["probability"])


def gamma(parameters):
    return RandomGamma(p=parameters["probability"])


def flip(parameters, axes=0):
    return RandomFlip(axes=axes, p=parameters["probability"])


def anisotropy(parameters, axes=0, downsampling=1):
    return RandomAnisotropy(
        axes=axes, downsampling=downsampling, scalars_only=True, p=parameters["probability"]
        )
