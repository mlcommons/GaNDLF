import ast
import traceback
from copy import deepcopy
from GANDLF.configuration.differential_privacy_config import DifferentialPrivacyConfig
from GANDLF.configuration.post_processing_config import PostProcessingConfig
from GANDLF.configuration.pre_processing_config import (
    HistogramMatchingConfig,
    PreProcessingConfig,
)
from GANDLF.data.post_process import postprocessing_after_reverse_one_hot_encoding
import numpy as np
import sys
from GANDLF.configuration.optimizer_config import OptimizerConfig, optimizer_dict_config
from GANDLF.configuration.patch_sampler_config import PatchSamplerConfig
from GANDLF.configuration.scheduler_config import (
    SchedulerConfig,
    schedulers_dict_config,
)
from GANDLF.configuration.utils import initialize_key, combine_models
from GANDLF.metrics import surface_distance_ids


def validate_loss_function(value) -> dict:
    if isinstance(value, dict):  # if this is a dict
        if len(value) > 0:  # only proceed if something is defined
            for key in value:  # iterate through all keys
                if key == "mse":
                    if (value[key] is None) or not ("reduction" in value[key]):
                        value[key] = {}
                        value[key]["reduction"] = "mean"
                else:
                    # use simple string for other functions - can be extended with parameters, if needed
                    value = key
    else:
        if value == "focal":
            value = {"focal": {}}
            value["focal"]["gamma"] = 2.0
            value["focal"]["size_average"] = True
        elif value == "mse":
            value = {"mse": {}}
            value["mse"]["reduction"] = "mean"

    return value


def validate_metrics(value) -> dict:
    if not isinstance(value, dict):
        temp_dict = {}
    else:
        temp_dict = value

    # initialize metrics dict
    for metric in value:
        # assigning a new variable because some metrics can be dicts, and we want to get the first key
        comparison_string = metric
        if isinstance(metric, dict):
            comparison_string = list(metric.keys())[0]
        # these metrics always need to be dicts
        if comparison_string in [
            "accuracy",
            "f1",
            "precision",
            "recall",
            "specificity",
            "iou",
        ]:
            if not isinstance(metric, dict):
                temp_dict[metric] = {}
            else:
                temp_dict[comparison_string] = metric
        elif not isinstance(metric, dict):
            temp_dict[metric] = None

        # special case for accuracy, precision, recall, and specificity; which could be dicts
        ## need to find a better way to do this
        if any(
            _ in comparison_string
            for _ in ["precision", "recall", "specificity", "accuracy", "f1"]
        ):
            if comparison_string != "classification_accuracy":
                temp_dict[comparison_string] = initialize_key(
                    temp_dict[comparison_string], "average", "weighted"
                )
                temp_dict[comparison_string] = initialize_key(
                    temp_dict[comparison_string], "multi_class", True
                )
                temp_dict[comparison_string] = initialize_key(
                    temp_dict[comparison_string], "mdmc_average", "samplewise"
                )
                temp_dict[comparison_string] = initialize_key(
                    temp_dict[comparison_string], "threshold", 0.5
                )
                if comparison_string == "accuracy":
                    temp_dict[comparison_string] = initialize_key(
                        temp_dict[comparison_string], "subset_accuracy", False
                    )
        elif "iou" in comparison_string:
            temp_dict["iou"] = initialize_key(
                temp_dict["iou"], "reduction", "elementwise_mean"
            )
            temp_dict["iou"] = initialize_key(temp_dict["iou"], "threshold", 0.5)
        elif comparison_string in surface_distance_ids:
            temp_dict[comparison_string] = initialize_key(
                temp_dict[comparison_string], "connectivity", 1
            )
            temp_dict[comparison_string] = initialize_key(
                temp_dict[comparison_string], "threshold", None
            )

    value = temp_dict
    return value


def validate_class_list(value):
    if isinstance(value, str):
        if ("||" in value) or ("&&" in value):
            # special case for multi-class computation - this needs to be handled during one-hot encoding mask construction
            print(
                "WARNING: This is a special case for multi-class computation, where different labels are processed together, `reverse_one_hot` will need mapping information to work correctly"
            )
            temp_class_list = value
            # we don't need the brackets
            temp_class_list = temp_class_list.replace("[", "")
            temp_class_list = temp_class_list.replace("]", "")
            value = temp_class_list.split(",")
        else:
            try:
                value = ast.literal_eval(value)
                return value
            except Exception as e:
                assert (
                    False
                ), f"Could not evaluate the `class_list` in `model`, Exception: {str(e)}, {traceback.format_exc()}"
                # logging.error(
                #     f"Could not evaluate the `class_list` in `model`, Exception: {str(e)}, {traceback.format_exc()}"
                # )
    return value


def validate_patch_size(patch_size, dimension) -> list:
    if isinstance(patch_size, int) or isinstance(patch_size, float):
        patch_size = [patch_size]
    if len(patch_size) == 1 and dimension is not None:
        actual_patch_size = []
        for _ in range(dimension):
            actual_patch_size.append(patch_size[0])
        patch_size = actual_patch_size
    if len(patch_size) == 2:  # 2d check
        # ensuring same size during torchio processing
        patch_size.append(1)
        if dimension is None:
            dimension = 2
    elif len(patch_size) == 3:  # 2d check
        if dimension is None:
            dimension = 3
    return [patch_size, dimension]


def validate_norm_type(norm_type, architecture):
    if norm_type is None or norm_type.lower() == "none":
        if not ("vgg" in architecture):
            raise ValueError(
                "Normalization type cannot be 'None' for non-VGG architectures"
            )
    return norm_type


def validate_parallel_compute_command(value):
    parallel_compute_command = value
    parallel_compute_command = parallel_compute_command.replace("'", "")
    parallel_compute_command = parallel_compute_command.replace('"', "")
    value = parallel_compute_command
    return value


def validate_scheduler(value, learning_rate, num_epochs):
    if isinstance(value, str):
        value = SchedulerConfig(type=value)
    # Find the scheduler_config class based on the type
    combine_scheduler_class = schedulers_dict_config[value.type]
    # Combine it with the SchedulerConfig class
    schedulerConfigCombine = combine_models(SchedulerConfig, combine_scheduler_class)
    combineScheduler = schedulerConfigCombine(**value.model_dump())
    value = SchedulerConfig(**combineScheduler.model_dump())

    if value.type == "triangular":
        if value.max_lr is None:
            value.max_lr = learning_rate

    if value.type in [
        "reduce_on_plateau",
        "reduce-on-plateau",
        "plateau",
        "exp_range",
        "triangular",
    ]:
        if value.min_lr is None:
            value.min_lr = learning_rate * 0.001

    if value.type in ["warmupcosineschedule", "wcs"]:
        value.warmup_steps = num_epochs * 0.1

    if hasattr(value, "step_size") and value.step_size is None:
        value.step_size = learning_rate / 5.0

    return value


def validate_optimizer(value):
    if isinstance(value, str):
        value = OptimizerConfig(type=value)

    combine_optimizer_class = optimizer_dict_config[value.type]
    # Combine it with the OptimizerConfig class
    optimizerConfigCombine = combine_models(OptimizerConfig, combine_optimizer_class)
    combineOptimizer = optimizerConfigCombine(**value.model_dump())
    value = OptimizerConfig(**combineOptimizer.model_dump())

    return value


def validate_data_preprocessing(value) -> dict:
    if not (value is None):
        # perform this only when pre-processing is defined
        if len(value) > 0:
            thresholdOrClip = False
            # this can be extended, as required
            thresholdOrClipDict = ["threshold", "clip", "clamp"]

            resize_requested = False
            temp_dict = deepcopy(value)
            for key in value:
                if key in ["resize", "resize_image", "resize_images", "resize_patch"]:
                    resize_requested = True

                if key in ["resample_min", "resample_minimum"]:
                    if "resolution" in value[key]:
                        resize_requested = True
                        resolution_temp = np.array(value[key]["resolution"])
                        if resolution_temp.size == 1:
                            temp_dict[key]["resolution"] = np.array(
                                [resolution_temp, resolution_temp]
                            ).tolist()
                    else:
                        temp_dict.pop(key)

            value = temp_dict

            if resize_requested and "resample" in value:
                for key in ["resize", "resize_image", "resize_images", "resize_patch"]:
                    if key in value:
                        value.pop(key)

                print(
                    "WARNING: Different 'resize' operations are ignored as 'resample' is defined under 'data_processing'",
                    file=sys.stderr,
                )

            # iterate through all keys
            for key in value:  # iterate through all keys
                if key in thresholdOrClipDict:
                    # we only allow one of threshold or clip to occur and not both
                    assert not (
                        thresholdOrClip
                    ), "Use only `threshold` or `clip`, not both"
                    thresholdOrClip = True
                    # initialize if nothing is present
                    if not (isinstance(value[key], dict)):
                        value[key] = {}

                    # if one of the required parameters is not present, initialize with lowest/highest possible values
                    # this ensures the absence of a field doesn't affect processing
                    # for threshold or clip, ensure min and max are defined
                    if not "min" in value[key]:
                        value[key]["min"] = sys.float_info.min
                    if not "max" in value[key]:
                        value[key]["max"] = sys.float_info.max

            key = "histogram_matching"
            if key in value:
                if value["histogram_matching"] is not False:
                    if not (isinstance(value["histogram_matching"], dict)):
                        value["histogram_matching"] = HistogramMatchingConfig()

            key = "histogram_equalization"
            if key in value:
                if value[key] is not False:
                    # if histogram equalization is enabled, call histogram_matching
                    value["histogram_matching"] = HistogramMatchingConfig()
            key = "adaptive_histogram_equalization"
            if key in value:
                if value[key] is not False:
                    # if histogram equalization is enabled, call histogram_matching
                    value["histogram_matching"] = HistogramMatchingConfig(
                        target="adaptive"
                    )

    pre_processing = PreProcessingConfig(**value)
    return pre_processing.model_dump(include={field for field in value.keys()})


def validate_data_postprocessing_after_reverse_one_hot_encoding(
    value, data_postprocessing
) -> list:
    temp_dict = deepcopy(value)
    for key in temp_dict:
        if key in postprocessing_after_reverse_one_hot_encoding:
            value[key] = data_postprocessing[key]
            data_postprocessing.pop(key)
    return [value, data_postprocessing]


def validate_patch_sampler(value):
    if isinstance(value, str):
        value = PatchSamplerConfig(type=value.lower())
    return value


def validate_data_augmentation(value, patch_size) -> dict:
    value["default_probability"] = value.get("default_probability", 0.5)
    if not (value is None):
        if len(value) > 0:  # only when augmentations are defined
            # special case for random swapping and elastic transformations - which takes a patch size for computation
            for key in ["swap", "elastic"]:
                if key in value:
                    value[key] = initialize_key(
                        value[key],
                        "patch_size",
                        np.round(np.array(patch_size) / 10).astype("int").tolist(),
                    )

            # special case for swap default initialization
            if "swap" in value:
                value["swap"] = initialize_key(value["swap"], "num_iterations", 100)

            # special case for affine default initialization
            if "affine" in value:
                value["affine"] = initialize_key(value["affine"], "scales", 0.1)
                value["affine"] = initialize_key(value["affine"], "degrees", 15)
                value["affine"] = initialize_key(value["affine"], "translation", 2)

            if "motion" in value:
                value["motion"] = initialize_key(value["motion"], "num_transforms", 2)
                value["motion"] = initialize_key(value["motion"], "degrees", 15)
                value["motion"] = initialize_key(value["motion"], "translation", 2)
                value["motion"] = initialize_key(
                    value["motion"], "interpolation", "linear"
                )

            # special case for random blur/noise - which takes a std-dev range
            for std_aug in ["blur", "noise_var"]:
                if std_aug in value:
                    value[std_aug] = initialize_key(value[std_aug], "std", None)
            for std_aug in ["noise"]:
                if std_aug in value:
                    value[std_aug] = initialize_key(value[std_aug], "std", [0, 1])

            # special case for random noise - which takes a mean range
            for mean_aug in ["noise", "noise_var"]:
                if mean_aug in value:
                    value[mean_aug] = initialize_key(value[mean_aug], "mean", 0)

            # special case for augmentations that need axis defined
            for axis_aug in ["flip", "anisotropic", "rotate_90", "rotate_180"]:
                if axis_aug in value:
                    value[axis_aug] = initialize_key(value[axis_aug], "axis", [0, 1, 2])

            # special case for colorjitter
            if "colorjitter" in value:
                value = initialize_key(value, "colorjitter", {})
                for key in ["brightness", "contrast", "saturation"]:
                    value["colorjitter"] = initialize_key(
                        value["colorjitter"], key, [0, 1]
                    )
                value["colorjitter"] = initialize_key(
                    value["colorjitter"], "hue", [-0.5, 0.5]
                )

            # Added HED augmentation in gandlf
            hed_augmentation_types = [
                "hed_transform",
                # "hed_transform_light",
                # "hed_transform_heavy",
            ]
            for augmentation_type in hed_augmentation_types:
                if augmentation_type in value:
                    value = initialize_key(value, "hed_transform", {})
                    ranges = [
                        "haematoxylin_bias_range",
                        "eosin_bias_range",
                        "dab_bias_range",
                        "haematoxylin_sigma_range",
                        "eosin_sigma_range",
                        "dab_sigma_range",
                    ]

                    default_range = (
                        [-0.1, 0.1]
                        if augmentation_type == "hed_transform"
                        else (
                            [-0.03, 0.03]
                            if augmentation_type == "hed_transform_light"
                            else [-0.95, 0.95]
                        )
                    )

                    for key in ranges:
                        value["hed_transform"] = initialize_key(
                            value["hed_transform"], key, default_range
                        )

                    value["hed_transform"] = initialize_key(
                        value["hed_transform"], "cutoff_range", [0, 1]
                    )

            # special case for anisotropic
            if "anisotropic" in value:
                if not ("downsampling" in value["anisotropic"]):
                    default_downsampling = 1.5
                else:
                    default_downsampling = value["anisotropic"]["downsampling"]

                initialize_downsampling = False
                if isinstance(default_downsampling, list):
                    if len(default_downsampling) != 2:
                        initialize_downsampling = True
                        print(
                            "WARNING: 'anisotropic' augmentation needs to be either a single number of a list of 2 numbers: https://torchio.readthedocs.io/transforms/augmentation.html?highlight=randomswap#torchio.transforms.RandomAnisotropy.",
                            file=sys.stderr,
                        )
                        default_downsampling = default_downsampling[0]  # only
                else:
                    initialize_downsampling = True

                if initialize_downsampling:
                    if default_downsampling < 1:
                        print(
                            "WARNING: 'anisotropic' augmentation needs the 'downsampling' parameter to be greater than 1, defaulting to 1.5.",
                            file=sys.stderr,
                        )
                        # default
                    value["anisotropic"]["downsampling"] = 1.5

            for key in value:
                if key != "default_probability":
                    value[key] = initialize_key(
                        value[key], "probability", value["default_probability"]
                    )
    return value


def validate_postprocessing(value):
    post_processing = PostProcessingConfig(**value)
    return post_processing.model_dump(include={field for field in value.keys()})


def validate_differential_privacy(value, batch_size):
    if value is None:
        return value
    if not isinstance(value, dict):
        print(
            "WARNING: Non dictionary value for the key: 'differential_privacy' was used, replacing with default valued dictionary."
        )
        value = DifferentialPrivacyConfig(physical_batch_size=batch_size).model_dump()
    # these are some defaults

    if value["physical_batch_size"] > batch_size:
        print(
            f"WARNING: The physical batch size {value['physical_batch_size']} is greater"
            f"than the batch size {batch_size}, setting the physical batch size to the batch size."
        )
    value["physical_batch_size"] = batch_size

    # these keys need to be parsed as floats, not strings
    for key in ["noise_multiplier", "max_grad_norm", "delta", "epsilon"]:
        if key in value:
            value[key] = float(value[key])

    return DifferentialPrivacyConfig(**value)
