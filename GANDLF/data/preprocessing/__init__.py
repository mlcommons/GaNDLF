import numpy as np
import torchio
from torchio.transforms import (
    Resample,
    Compose,
    Pad,
)

from .crop_zero_planes import CropExternalZeroplanes
from .non_zero_normalize import NonZeroNormalizeOnMaskedRegion
from .threshold_and_clip import (
    threshold_transform,
    clip_transform,
)
from .normalize_rgb import (
    normalize_by_val_transform,
    normalize_imagenet_transform,
    normalize_standardize_transform,
    normalize_div_by_255_transform,
)
from .resample_minimum import Resample_Minimum

from torchio.transforms import (
    ZNormalization,
    ToCanonical,
    Crop,
    CropOrPad,
    Resize,
    Resample,
)


def positive_voxel_mask(image):
    return image > 0


def nonzero_voxel_mask(image):
    return image != 0


def to_canonical_transform(parameters):
    return ToCanonical()


def crop_transform(patch_size):
    return Crop(patch_size)


def centercrop_transform(patch_size):
    return CropOrPad(target_shape=patch_size)


# defining dict for pre-processing - key is the string and the value is the transform object
global_preprocessing_dict = {
    "to_canonical": to_canonical_transform,
    "threshold": threshold_transform,
    "clip": clip_transform,
    "clamp": clip_transform,
    "crop_external_zero_planes": CropExternalZeroplanes,
    "crop": crop_transform,
    "centercrop": centercrop_transform,
    "normalize_by_val": normalize_by_val_transform,
    "normalize_imagenet": normalize_imagenet_transform(),
    "normalize_standardize": normalize_standardize_transform(),
    "normalize_div_by_255": normalize_div_by_255_transform(),
    "normalize": ZNormalization(),
    "normalize_positive": ZNormalization(masking_method=positive_voxel_mask),
    "normalize_nonZero": ZNormalization(masking_method=nonzero_voxel_mask),
    "normalize_nonzero": ZNormalization(masking_method=nonzero_voxel_mask),
    "normalize_nonZero_masked": NonZeroNormalizeOnMaskedRegion(),
    "normalize_nonzero_masked": NonZeroNormalizeOnMaskedRegion(),
}


def get_transforms_for_preprocessing(parameters, current_transformations, train_mode, apply_zero_crop):

    preprocessing = parameters["data_preprocessing"]
    # first, we want to do thresholding, followed by clipping, if it is present - required for inference as well
    normalize_to_apply = None
    if not (preprocessing is None):
        # go through preprocessing in the order they are specified
        for preprocess in preprocessing:
            preprocess_lower = preprocess.lower()
            # special check for resample
            if preprocess_lower == "resize":
                resize_values = tuple(preprocessing["resize"])
                current_transformations.append(torchio.Resize(resize_values))
            elif preprocess_lower == "resize_patch":
                resize_values = tuple(preprocessing["resize_patch"])
                current_transformations.append(torchio.Resize(resize_values))
            elif preprocess_lower == "resample":
                if "resolution" in preprocessing[preprocess_lower]:
                    # Need to take a look here
                    resample_values = np.array(
                        preprocessing[preprocess_lower]["resolution"]
                    )
                    if len(resample_values) == 2:
                        resample_values = tuple(
                            np.append(
                                np.array(preprocessing[preprocess_lower]["resolution"]),
                                1,
                            )
                        )
                    current_transformations.append(Resample(resample_values))
            elif preprocess_lower in ["resample_minimum", "resample_min"]:
                if "resolution" in preprocessing[preprocess_lower]:
                    current_transformations.append(
                        Resample_Minimum(
                            np.array(preprocessing[preprocess_lower]["resolution"])
                        )
                    )
            # normalize should be applied at the end
            elif "normalize" in preprocess_lower:
                if normalize_to_apply is None:
                    normalize_to_apply = global_preprocessing_dict[preprocess_lower]
            # preprocessing routines that we only want for training
            elif preprocess_lower in ["crop_external_zero_planes"]:
                if train_mode or apply_zero_crop:
                    current_transformations.append(
                        global_preprocessing_dict["crop_external_zero_planes"](
                            patch_size=parameters["patch_size"]
                        )
                    )
            # everything else is taken in the order passed by user
            elif preprocess_lower in global_preprocessing_dict:
                current_transformations.append(
                    global_preprocessing_dict[preprocess_lower](
                        preprocessing[preprocess]
                    )
                )

    # normalization type is applied at the end
    if normalize_to_apply is not None:
        current_transformations.append(normalize_to_apply)

    # compose the transformations
    transforms_to_apply = None
    if current_transformations:
        transforms_to_apply = Compose(current_transformations)
    
    return transforms_to_apply