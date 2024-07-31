# import logging
import traceback
from typing import Optional, Union
import sys, yaml, ast
import numpy as np
from copy import deepcopy

from .utils import version_check
from GANDLF.data.post_process import postprocessing_after_reverse_one_hot_encoding

from GANDLF.metrics import surface_distance_ids
from importlib.metadata import version

## dictionary to define defaults for appropriate options, which are evaluated
parameter_defaults = {
    "weighted_loss": False,  # whether weighted loss is to be used or not
    "verbose": False,  # general application verbosity
    "q_verbose": False,  # queue construction verbosity
    "medcam_enabled": False,  # interpretability via medcam
    "save_training": False,  # save outputs during training
    "save_output": False,  # save outputs during validation/testing
    "in_memory": False,  # pin data to cpu memory
    "pin_memory_dataloader": False,  # pin data to gpu memory
    "scaling_factor": 1,  # scaling factor for regression problems
    "q_max_length": 100,  # the max length of queue
    "q_samples_per_volume": 10,  # number of samples per volume
    "q_num_workers": 4,  # number of worker threads to use
    "num_epochs": 100,  # total number of epochs to train
    "patience": 100,  # number of epochs to wait for performance improvement
    "batch_size": 1,  # default batch size of training
    "learning_rate": 0.001,  # default learning rate
    "clip_grad": None,  # clip_gradient value
    "track_memory_usage": False,  # default memory tracking
    "memory_save_mode": False,  # default memory saving, if enabled, resize/resample will save files to disk
    "print_rgb_label_warning": True,  # print rgb label warning
    "data_postprocessing": {},  # default data postprocessing
    "grid_aggregator_overlap": "crop",  # default grid aggregator overlap strategy
    "determinism": False,  # using deterministic version of computation
    "previous_parameters": None,  # previous parameters to be used for resuming training and perform sanity checking
}

## dictionary to define string defaults for appropriate options
parameter_defaults_string = {
    "optimizer": "adam",  # the optimizer
    "scheduler": "triangle_modified",  # the default scheduler
    "clip_mode": None,  # default clip mode
}


def initialize_parameter(
    params: dict,
    parameter_to_initialize: str,
    value: Optional[Union[str, list, int, dict]] = None,
    evaluate: Optional[bool] = True,
) -> dict:
    """
    This function will initialize the parameter in the parameters dict to the value if it is absent.

    Args:
        params (dict): The parameter dictionary.
        parameter_to_initialize (str): The parameter to initialize.
        value (Optional[Union[str, list, int, dict]], optional): The value to initialize. Defaults to None.
        evaluate (Optional[bool], optional): Whether to evaluate the value. Defaults to True.

    Returns:
        dict: The parameter dictionary.
    """
    if parameter_to_initialize in params:
        if evaluate:
            if isinstance(params[parameter_to_initialize], str):
                if params[parameter_to_initialize].lower() == "none":
                    params[parameter_to_initialize] = ast.literal_eval(
                        params[parameter_to_initialize]
                    )
    else:
        print(
            "WARNING: Initializing '" + parameter_to_initialize + "' as " + str(value)
        )
        params[parameter_to_initialize] = value

    return params


def initialize_key(
    parameters: dict, key: str, value: Optional[Union[str, float, list, dict]] = None
) -> dict:
    """
    This function initializes a key in the parameters dictionary to a value if it is absent.

    Args:
        parameters (dict): The parameter dictionary.
        key (str): The key to initialize.
        value (Optional[Union[str, float, list, dict]], optional): The value to initialize. Defaults to None.

    Returns:
        dict: The parameter dictionary.
    """
    if parameters is None:
        parameters = {}
    if key in parameters:
        if parameters[key] is not None:
            if isinstance(parameters[key], dict):
                # if key is present but not defined
                if len(parameters[key]) == 0:
                    parameters[key] = value
    else:
        parameters[key] = value  # if key is absent

    return parameters


def _parseConfig(
    config_file_path: Union[str, dict], version_check_flag: bool = True
) -> None:
    """
    This function parses the configuration file and returns a dictionary of parameters.

    Args:
        config_file_path (Union[str, dict]): The filename of the configuration file.
        version_check_flag (bool, optional): Whether to check the version in configuration file. Defaults to True.

    Returns:
        dict: The parameter dictionary.
    """
    params = config_file_path
    if not isinstance(config_file_path, dict):
        params = yaml.safe_load(open(config_file_path, "r"))

    if version_check_flag:  # this is only to be used for testing
        assert (
            "version" in params
        ), "The 'version' key needs to be defined in config with 'minimum' and 'maximum' fields to determine the compatibility of configuration with code base"
        version_check(params["version"], version_to_check=version("GANDLF"))

    if "patch_size" in params:
        # duplicate patch size if it is an int or float
        if isinstance(params["patch_size"], int) or isinstance(
            params["patch_size"], float
        ):
            params["patch_size"] = [params["patch_size"]]
        # in case someone decides to pass a single value list
        if len(params["patch_size"]) == 1:
            actual_patch_size = []
            for _ in range(params["model"]["dimension"]):
                actual_patch_size.append(params["patch_size"][0])
            params["patch_size"] = actual_patch_size

        # parse patch size as needed for computations
        if len(params["patch_size"]) == 2:  # 2d check
            # ensuring same size during torchio processing
            params["patch_size"].append(1)
            if "dimension" not in params["model"]:
                params["model"]["dimension"] = 2
        elif len(params["patch_size"]) == 3:  # 2d check
            if "dimension" not in params["model"]:
                params["model"]["dimension"] = 3
    assert "patch_size" in params, "Patch size needs to be defined in the config file"

    if "resize" in params:
        print(
            "WARNING: 'resize' should be defined under 'data_processing', this will be skipped",
            file=sys.stderr,
        )

    assert "modality" in params, "'modality' needs to be defined in the config file"
    params["modality"] = params["modality"].lower()
    assert params["modality"] in [
        "rad",
        "histo",
        "path",
    ], "Modality should be either 'rad' or 'path'"

    assert (
        "loss_function" in params
    ), "'loss_function' needs to be defined in the config file"
    if "loss_function" in params:
        # check if user has passed a dict
        if isinstance(params["loss_function"], dict):  # if this is a dict
            if len(params["loss_function"]) > 0:  # only proceed if something is defined
                for key in params["loss_function"]:  # iterate through all keys
                    if key == "mse":
                        if (params["loss_function"][key] is None) or not (
                            "reduction" in params["loss_function"][key]
                        ):
                            params["loss_function"][key] = {}
                            params["loss_function"][key]["reduction"] = "mean"
                    else:
                        # use simple string for other functions - can be extended with parameters, if needed
                        params["loss_function"] = key
        else:
            # check if user has passed a single string
            if params["loss_function"] == "mse":
                params["loss_function"] = {}
                params["loss_function"]["mse"] = {}
                params["loss_function"]["mse"]["reduction"] = "mean"
            elif params["loss_function"] == "focal":
                params["loss_function"] = {}
                params["loss_function"]["focal"] = {}
                params["loss_function"]["focal"]["gamma"] = 2.0
                params["loss_function"]["focal"]["size_average"] = True

    assert "metrics" in params, "'metrics' needs to be defined in the config file"
    if "metrics" in params:
        if not isinstance(params["metrics"], dict):
            temp_dict = {}
        else:
            temp_dict = params["metrics"]

        # initialize metrics dict
        for metric in params["metrics"]:
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

        params["metrics"] = temp_dict

    # this is NOT a required parameter - a user should be able to train with NO augmentations
    params = initialize_key(params, "data_augmentation", {})
    # for all others, ensure probability is present
    params["data_augmentation"]["default_probability"] = params[
        "data_augmentation"
    ].get("default_probability", 0.5)

    if not (params["data_augmentation"] is None):
        if len(params["data_augmentation"]) > 0:  # only when augmentations are defined
            # special case for random swapping and elastic transformations - which takes a patch size for computation
            for key in ["swap", "elastic"]:
                if key in params["data_augmentation"]:
                    params["data_augmentation"][key] = initialize_key(
                        params["data_augmentation"][key],
                        "patch_size",
                        np.round(np.array(params["patch_size"]) / 10)
                        .astype("int")
                        .tolist(),
                    )

            # special case for swap default initialization
            if "swap" in params["data_augmentation"]:
                params["data_augmentation"]["swap"] = initialize_key(
                    params["data_augmentation"]["swap"], "num_iterations", 100
                )

            # special case for affine default initialization
            if "affine" in params["data_augmentation"]:
                params["data_augmentation"]["affine"] = initialize_key(
                    params["data_augmentation"]["affine"], "scales", 0.1
                )
                params["data_augmentation"]["affine"] = initialize_key(
                    params["data_augmentation"]["affine"], "degrees", 15
                )
                params["data_augmentation"]["affine"] = initialize_key(
                    params["data_augmentation"]["affine"], "translation", 2
                )

            if "motion" in params["data_augmentation"]:
                params["data_augmentation"]["motion"] = initialize_key(
                    params["data_augmentation"]["motion"], "num_transforms", 2
                )
                params["data_augmentation"]["motion"] = initialize_key(
                    params["data_augmentation"]["motion"], "degrees", 15
                )
                params["data_augmentation"]["motion"] = initialize_key(
                    params["data_augmentation"]["motion"], "translation", 2
                )
                params["data_augmentation"]["motion"] = initialize_key(
                    params["data_augmentation"]["motion"], "interpolation", "linear"
                )

            # special case for random blur/noise - which takes a std-dev range
            for std_aug in ["blur", "noise_var"]:
                if std_aug in params["data_augmentation"]:
                    params["data_augmentation"][std_aug] = initialize_key(
                        params["data_augmentation"][std_aug], "std", None
                    )
            for std_aug in ["noise"]:
                if std_aug in params["data_augmentation"]:
                    params["data_augmentation"][std_aug] = initialize_key(
                        params["data_augmentation"][std_aug], "std", [0, 1]
                    )

            # special case for random noise - which takes a mean range
            for mean_aug in ["noise", "noise_var"]:
                if mean_aug in params["data_augmentation"]:
                    params["data_augmentation"][mean_aug] = initialize_key(
                        params["data_augmentation"][mean_aug], "mean", 0
                    )

            # special case for augmentations that need axis defined
            for axis_aug in ["flip", "anisotropic", "rotate_90", "rotate_180"]:
                if axis_aug in params["data_augmentation"]:
                    params["data_augmentation"][axis_aug] = initialize_key(
                        params["data_augmentation"][axis_aug], "axis", [0, 1, 2]
                    )

            # special case for colorjitter
            if "colorjitter" in params["data_augmentation"]:
                params["data_augmentation"] = initialize_key(
                    params["data_augmentation"], "colorjitter", {}
                )
                for key in ["brightness", "contrast", "saturation"]:
                    params["data_augmentation"]["colorjitter"] = initialize_key(
                        params["data_augmentation"]["colorjitter"], key, [0, 1]
                    )
                params["data_augmentation"]["colorjitter"] = initialize_key(
                    params["data_augmentation"]["colorjitter"], "hue", [-0.5, 0.5]
                )

            # Added HED augmentation in gandlf
            hed_augmentation_types = [
                "hed_transform",
                # "hed_transform_light",
                # "hed_transform_heavy",
            ]
            for augmentation_type in hed_augmentation_types:
                if augmentation_type in params["data_augmentation"]:
                    params["data_augmentation"] = initialize_key(
                        params["data_augmentation"], "hed_transform", {}
                    )
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
                        params["data_augmentation"]["hed_transform"] = initialize_key(
                            params["data_augmentation"]["hed_transform"],
                            key,
                            default_range,
                        )

                    params["data_augmentation"]["hed_transform"] = initialize_key(
                        params["data_augmentation"]["hed_transform"],
                        "cutoff_range",
                        [0, 1],
                    )

            # special case for anisotropic
            if "anisotropic" in params["data_augmentation"]:
                if not ("downsampling" in params["data_augmentation"]["anisotropic"]):
                    default_downsampling = 1.5
                else:
                    default_downsampling = params["data_augmentation"]["anisotropic"][
                        "downsampling"
                    ]

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
                    params["data_augmentation"]["anisotropic"]["downsampling"] = 1.5

            for key in params["data_augmentation"]:
                if key != "default_probability":
                    params["data_augmentation"][key] = initialize_key(
                        params["data_augmentation"][key],
                        "probability",
                        params["data_augmentation"]["default_probability"],
                    )

    # this is NOT a required parameter - a user should be able to train with NO built-in pre-processing
    params = initialize_key(params, "data_preprocessing", {})
    if not (params["data_preprocessing"] is None):
        # perform this only when pre-processing is defined
        if len(params["data_preprocessing"]) > 0:
            thresholdOrClip = False
            # this can be extended, as required
            thresholdOrClipDict = ["threshold", "clip", "clamp"]

            resize_requested = False
            temp_dict = deepcopy(params["data_preprocessing"])
            for key in params["data_preprocessing"]:
                if key in ["resize", "resize_image", "resize_images", "resize_patch"]:
                    resize_requested = True

                if key in ["resample_min", "resample_minimum"]:
                    if "resolution" in params["data_preprocessing"][key]:
                        resize_requested = True
                        resolution_temp = np.array(
                            params["data_preprocessing"][key]["resolution"]
                        )
                        if resolution_temp.size == 1:
                            temp_dict[key]["resolution"] = np.array(
                                [resolution_temp, resolution_temp]
                            ).tolist()
                    else:
                        temp_dict.pop(key)

            params["data_preprocessing"] = temp_dict

            if resize_requested and "resample" in params["data_preprocessing"]:
                for key in ["resize", "resize_image", "resize_images", "resize_patch"]:
                    if key in params["data_preprocessing"]:
                        params["data_preprocessing"].pop(key)

                print(
                    "WARNING: Different 'resize' operations are ignored as 'resample' is defined under 'data_processing'",
                    file=sys.stderr,
                )

            # iterate through all keys
            for key in params["data_preprocessing"]:  # iterate through all keys
                if key in thresholdOrClipDict:
                    # we only allow one of threshold or clip to occur and not both
                    assert not (
                        thresholdOrClip
                    ), "Use only `threshold` or `clip`, not both"
                    thresholdOrClip = True
                    # initialize if nothing is present
                    if not (isinstance(params["data_preprocessing"][key], dict)):
                        params["data_preprocessing"][key] = {}

                    # if one of the required parameters is not present, initialize with lowest/highest possible values
                    # this ensures the absence of a field doesn't affect processing
                    # for threshold or clip, ensure min and max are defined
                    if not "min" in params["data_preprocessing"][key]:
                        params["data_preprocessing"][key]["min"] = sys.float_info.min
                    if not "max" in params["data_preprocessing"][key]:
                        params["data_preprocessing"][key]["max"] = sys.float_info.max

                if key == "histogram_matching":
                    if params["data_preprocessing"][key] is not False:
                        if not (isinstance(params["data_preprocessing"][key], dict)):
                            params["data_preprocessing"][key] = {}

                if key == "histogram_equalization":
                    if params["data_preprocessing"][key] is not False:
                        # if histogram equalization is enabled, call histogram_matching
                        params["data_preprocessing"]["histogram_matching"] = {}

                if key == "adaptive_histogram_equalization":
                    if params["data_preprocessing"][key] is not False:
                        # if histogram equalization is enabled, call histogram_matching
                        params["data_preprocessing"]["histogram_matching"] = {
                            "target": "adaptive"
                        }

    # this is NOT a required parameter - a user should be able to train with NO built-in post-processing
    params = initialize_key(params, "data_postprocessing", {})
    params = initialize_key(
        params, "data_postprocessing_after_reverse_one_hot_encoding", {}
    )
    temp_dict = deepcopy(params["data_postprocessing"])
    for key in temp_dict:
        if key in postprocessing_after_reverse_one_hot_encoding:
            params["data_postprocessing_after_reverse_one_hot_encoding"][key] = params[
                "data_postprocessing"
            ][key]
            params["data_postprocessing"].pop(key)

    if "model" in params:
        assert isinstance(
            params["model"], dict
        ), "The 'model' parameter needs to be populated as a dictionary"
        assert (
            len(params["model"]) > 0
        ), "The 'model' parameter needs to be populated as a dictionary and should have all properties present"
        assert (
            "architecture" in params["model"]
        ), "The 'model' parameter needs 'architecture' to be defined"
        assert (
            "final_layer" in params["model"]
        ), "The 'model' parameter needs 'final_layer' to be defined"
        assert (
            "dimension" in params["model"]
        ), "The 'model' parameter needs 'dimension' to be defined"

        if "amp" in params["model"]:
            pass
        else:
            print("NOT using Mixed Precision Training")
            params["model"]["amp"] = False

        if "norm_type" in params["model"]:
            if (
                params["model"]["norm_type"] == None
                or params["model"]["norm_type"].lower() == "none"
            ):
                if not ("vgg" in params["model"]["architecture"]):
                    raise ValueError(
                        "Normalization type cannot be 'None' for non-VGG architectures"
                    )
        else:
            print("WARNING: Initializing 'norm_type' as 'batch'", flush=True)
            params["model"]["norm_type"] = "batch"

        if not ("base_filters" in params["model"]):
            base_filters = 32
            params["model"]["base_filters"] = base_filters
            print("Using default 'base_filters' in 'model': ", base_filters)
        if not ("class_list" in params["model"]):
            params["model"]["class_list"] = []  # ensure that this is initialized
        if not ("ignore_label_validation" in params["model"]):
            params["model"]["ignore_label_validation"] = None
        if "batch_norm" in params["model"]:
            print(
                "WARNING: 'batch_norm' is no longer supported, please use 'norm_type' in 'model' instead",
                flush=True,
            )
        params["model"]["print_summary"] = params["model"].get("print_summary", True)

        channel_keys_to_check = ["n_channels", "channels", "model_channels"]
        for key in channel_keys_to_check:
            if key in params["model"]:
                params["model"]["num_channels"] = params["model"][key]
                break

        # initialize model type for processing: if not defined, default to torch
        if not ("type" in params["model"]):
            params["model"]["type"] = "torch"

        # initialize openvino model data type for processing: if not defined, default to FP32
        if not ("data_type" in params["model"]):
            params["model"]["data_type"] = "FP32"

        # set default save strategy for model
        if not ("save_at_every_epoch" in params["model"]):
            params["model"]["save_at_every_epoch"] = False

        if params["model"]["save_at_every_epoch"]:
            print(
                "WARNING: 'save_at_every_epoch' will result in TREMENDOUS storage usage; use at your own risk."
            )

    if isinstance(params["model"]["class_list"], str):
        if ("||" in params["model"]["class_list"]) or (
            "&&" in params["model"]["class_list"]
        ):
            # special case for multi-class computation - this needs to be handled during one-hot encoding mask construction
            print(
                "WARNING: This is a special case for multi-class computation, where different labels are processed together, `reverse_one_hot` will need mapping information to work correctly"
            )
            temp_classList = params["model"]["class_list"]
            # we don't need the brackets
            temp_classList = temp_classList.replace("[", "")
            temp_classList = temp_classList.replace("]", "")
            params["model"]["class_list"] = temp_classList.split(",")
        else:
            try:
                params["model"]["class_list"] = eval(params["model"]["class_list"])
            except Exception as e:
                ## todo: ensure logging captures assertion errors
                assert (
                    False
                ), f"Could not evaluate the `class_list` in `model`, Exception: {str(e)}, {traceback.format_exc()}"
                # logging.error(
                #     f"Could not evaluate the `class_list` in `model`, Exception: {str(e)}, {traceback.format_exc()}"
                # )

    assert (
        "nested_training" in params
    ), "The parameter 'nested_training' needs to be defined"
    # initialize defaults for nested training
    params["nested_training"]["stratified"] = params["nested_training"].get(
        "stratified", False
    )
    params["nested_training"]["stratified"] = params["nested_training"].get(
        "proportional", params["nested_training"]["stratified"]
    )
    params["nested_training"]["testing"] = params["nested_training"].get("testing", -5)
    params["nested_training"]["validation"] = params["nested_training"].get(
        "validation", -5
    )

    parallel_compute_command = ""
    if "parallel_compute_command" in params:
        parallel_compute_command = params["parallel_compute_command"]
        parallel_compute_command = parallel_compute_command.replace("'", "")
        parallel_compute_command = parallel_compute_command.replace('"', "")
    params["parallel_compute_command"] = parallel_compute_command

    if "opt" in params:
        print("DeprecationWarning: 'opt' has been superseded by 'optimizer'")
        params["optimizer"] = params["opt"]

    # initialize defaults for patch sampler
    temp_patch_sampler_dict = {
        "type": "uniform",
        "enable_padding": False,
        "padding_mode": "symmetric",
        "biased_sampling": False,
    }
    # check if patch_sampler is defined in the config
    if "patch_sampler" in params:
        # if "patch_sampler" is a string, then it is the type of sampler
        if isinstance(params["patch_sampler"], str):
            print(
                "WARNING: Defining 'patch_sampler' as a string will be deprecated in a future release, please use a dictionary instead"
            )
            temp_patch_sampler_dict["type"] = params["patch_sampler"].lower()
        elif isinstance(params["patch_sampler"], dict):
            # dict requires special handling
            for key in params["patch_sampler"]:
                temp_patch_sampler_dict[key] = params["patch_sampler"][key]

    # now assign the dict back to the params
    params["patch_sampler"] = temp_patch_sampler_dict
    del temp_patch_sampler_dict

    # define defaults
    for current_parameter in parameter_defaults:
        params = initialize_parameter(
            params, current_parameter, parameter_defaults[current_parameter], True
        )

    for current_parameter in parameter_defaults_string:
        params = initialize_parameter(
            params,
            current_parameter,
            parameter_defaults_string[current_parameter],
            False,
        )

    # ensure that the scheduler and optimizer are dicts
    if isinstance(params["scheduler"], str):
        temp_dict = {}
        temp_dict["type"] = params["scheduler"]
        params["scheduler"] = temp_dict

    if not ("step_size" in params["scheduler"]):
        params["scheduler"]["step_size"] = params["learning_rate"] / 5.0
        print(
            "WARNING: Setting default step_size to:", params["scheduler"]["step_size"]
        )

    # initialize default optimizer
    params["optimizer"] = params.get("optimizer", {})
    if isinstance(params["optimizer"], str):
        temp_dict = {}
        temp_dict["type"] = params["optimizer"]
        params["optimizer"] = temp_dict

    # initialize defaults for inference mechanism
    inference_mechanism = {"grid_aggregator_overlap": "crop", "patch_overlap": 0}
    initialize_inference_mechanism = False
    if not ("inference_mechanism" in params):
        initialize_inference_mechanism = True
    elif not (isinstance(params["inference_mechanism"], dict)):
        initialize_inference_mechanism = True
    else:
        for key in inference_mechanism:
            if not (key in params["inference_mechanism"]):
                params["inference_mechanism"][key] = inference_mechanism[key]

    if initialize_inference_mechanism:
        params["inference_mechanism"] = inference_mechanism

    return params


def ConfigManager(
    config_file_path: Union[str, dict], version_check_flag: bool = True
) -> None:
    """
    This function parses the configuration file and returns a dictionary of parameters.

    Args:
        config_file_path (Union[str, dict]): The filename of the configuration file.
        version_check_flag (bool, optional): Whether to check the version in configuration file. Defaults to True.

    Returns:
        dict: The parameter dictionary.
    """
    try:
        return _parseConfig(config_file_path, version_check_flag)
    except Exception as e:
        ## todo: ensure logging captures assertion errors
        assert (
            False
        ), f"Config parsing failed: {config_file_path=}, {version_check_flag=}, Exception: {str(e)}, {traceback.format_exc()}"
        # logging.error(
        #     f"gandlf config parsing failed: {config_file_path=}, {version_check_flag=}, Exception: {str(e)}, {traceback.format_exc()}"
        # )
        # raise
