import numpy as np

from torchio.transforms import (
    OneOf,
    RandomMotion,
    RandomGhosting,
    RandomSpike,
    RandomAffine,
    RandomElasticDeformation,
    RandomBiasField,
    RandomSwap,
    RandomNoise,
    RandomAnisotropy,
    RandomFlip,
    RandomGamma,
)
from .blur_enhanced import RandomBlurEnhanced
from .noise_enhanced import RandomNoiseEnhanced


## define helper functions to create transforms
## todo: ability to change interpolation type from config file
## todo: ability to change the dimensionality according to the config file
# define individual functions/lambdas for augmentations to handle properties
def mri_artifact(parameters):
    return OneOf(
        {RandomGhosting(): 0.5, RandomSpike(): 0.5},
        p=parameters["probability"],
    )


def motion(parameters):
    return RandomMotion(
        degrees=parameters["degrees"],
        translation=parameters["translation"],
        num_transforms=parameters["num_transforms"],
        image_interpolation=parameters["interpolation"],
        p=parameters["probability"],
    )


def affine(parameters):
    return RandomAffine(
        scales=parameters["scales"],
        degrees=parameters["degrees"],
        translation=parameters["translation"],
        p=parameters["probability"],
    )


def elastic(parameters):
    # define defaults
    parameters["num_control_points"] = parameters.get("num_control_points", None)
    parameters["max_displacement"] = parameters.get("max_displacement", None)
    parameters["locked_borders"] = parameters.get("locked_borders", 2)
    assert (
        "patch_size" in parameters
    ), "'patch_size' must be defined for elastic deformation"

    if parameters["num_control_points"] is None:
        # define the control points and swap axes for augmentation
        parameters["num_control_points"] = []
        for _, n in enumerate(parameters["patch_size"]):
            parameters["num_control_points"].append(max(n, 5))  # always at least have 5

    if parameters["max_displacement"] is None:
        parameters["max_displacement"] = np.divide(parameters["num_control_points"], 10)
        if parameters["num_control_points"][-1] == 1:
            # ensure maximum displacement is never greater than patch size
            parameters["max_displacement"][-1] = 0.1
        parameters["max_displacement"] = parameters["max_displacement"].tolist()

    return RandomElasticDeformation(
        num_control_points=parameters["num_control_points"],
        max_displacement=parameters["max_displacement"],
        locked_borders=parameters["locked_borders"],
        p=parameters["probability"],
    )


def swap(parameters):
    return RandomSwap(
        patch_size=parameters["patch_size"],
        num_iterations=100,
        p=parameters["probability"],
    )


def bias(parameters):
    return RandomBiasField(coefficients=0.5, order=3, p=parameters["probability"])


def blur(parameters):
    return RandomBlurEnhanced(std=parameters["std"], p=parameters["probability"])


def noise(parameters):
    return RandomNoise(
        mean=parameters["mean"], std=parameters["std"], p=parameters["probability"]
    )


def noise_var(parameters):
    return RandomNoiseEnhanced(
        mean=parameters["mean"], std=parameters["std"], p=parameters["probability"]
    )


def gamma(parameters):
    return RandomGamma(p=parameters["probability"])


def flip(parameters):
    return RandomFlip(axes=parameters["axis"], p=parameters["probability"])


def anisotropy(parameters):
    return RandomAnisotropy(
        axes=parameters["axis"],
        downsampling=parameters["downsampling"],
        scalars_only=True,
        p=parameters["probability"],
    )
