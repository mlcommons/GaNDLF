import sys, yaml, ast, pkg_resources
import numpy as np

from .utils import version_check

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
    "enable_padding": False,  # if padding needs to be done when "patch_sampler" is "label"
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
    "print_rgb_label_warning": True,  # default memory tracking
}

## dictionary to define string defaults for appropriate options
parameter_defaults_string = {
    "optimizer": "adam",  # the optimizer
    "patch_sampler": "uniform",  # type of sampling strategy
    "scheduler": "triangle_modified",  # the default scheduler
    "loss_function": "dc",  # default loss
    "clip_mode": None,  # default clip mode
}


def initialize_parameter(params, parameter_to_initialize, value=None, evaluate=True):
    """
    Initializes the specified parameter with supplied value

    Args:
        params (dict): The parameter dictionary.
        parameter_to_initialize (str): The parameter to initialize.
        value ((Union[str, list, int]), optional): The value to initialize. Defaults to None.
        evaluate (bool, optional): String evaluate. Defaults to True.

    Returns:
        [type]: [description]
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


def initialize_key(parameters, key, value=None):
    """
    This function will initialize the key in the parameters dict to 'None' if it is absent or length is zero.

    Args:
        parameters (dict): The parameter dictionary.
        key (str): The parameter to initialize.
        value (n.a.): The value to initialize.

    Returns:
        dict: The final parameter dictionary.
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


def parseConfig(config_file_path, version_check_flag=True):
    """
    This function parses the configuration file and returns a dictionary of parameters.

    Args:
        config_file_path (str): The filename of the configuration file.
        version_check_flag (bool, optional): Whether to check the version in configuration file. Defaults to True.

    Returns:
        dict: The parameter dictionary.
    """
    with open(config_file_path) as f:
        params = yaml.safe_load(f)

    if version_check_flag:  # this is only to be used for testing
        if not ("version" in params):
            sys.exit(
                "The 'version' key needs to be defined in config with 'minimum' and 'maximum' fields to determine the compatibility of configuration with code base"
            )
        else:
            version_check(
                params["version"],
                version_to_check=pkg_resources.require("GANDLF")[0].version,
            )

    if "patch_size" in params:
        if len(params["patch_size"]) == 2:  # 2d check
            # ensuring same size during torchio processing
            params["patch_size"].append(1)
    else:
        sys.exit(
            "The 'patch_size' parameter needs to be present in the configuration file"
        )

    if "resize" in params:
        print(
            "WARNING: 'resize' should be defined under 'data_processing', this will be skipped",
            file=sys.stderr,
        )

    if "modality" in params:
        modality = str(params["modality"])
        if modality.lower() == "rad":
            pass
        elif modality.lower() == "path":
            pass
        else:
            sys.exit(
                "The 'modality' should be set to either 'rad' or 'path'. Please check for spelling errors and it should be set to either of the two given options."
            )

    if "loss_function" in params:
        defineDefaultLoss = False
        # check if user has passed a dict
        if isinstance(params["loss_function"], dict):  # if this is a dict
            if len(params["loss_function"]) > 0:  # only proceed if something is defined
                for key in params["loss_function"]:  # iterate through all keys
                    if key == "mse":
                        if (params["loss_function"][key] == None) or not (
                            "reduction" in params["loss_function"][key]
                        ):
                            params["loss_function"][key] = {}
                            params["loss_function"][key]["reduction"] = "mean"
                    else:
                        # use simple string for other functions - can be extended with parameters, if needed
                        params["loss_function"] = key
            else:
                defineDefaultLoss = True
        else:
            # check if user has passed a single string
            if params["loss_function"] == "mse":
                params["loss_function"] = {}
                params["loss_function"]["mse"] = {}
                params["loss_function"]["mse"]["reduction"] = "mean"
    else:
        defineDefaultLoss = True
    if defineDefaultLoss == True:
        loss_function = "dc"
        print("Using default loss_function: ", loss_function)
    else:
        loss_function = params["loss_function"]
    params["loss_function"] = loss_function

    if "metrics" in params:
        if not isinstance(params["metrics"], dict):
            temp_dict = {}
        else:
            temp_dict = params["metrics"]

        # initialize metrics dict
        for metric in params["metrics"]:
            if not isinstance(metric, dict):
                temp_dict[metric] = None
            # special case for accuracy, precision, and recall; which could be dicts
            ## need to find a better way to do this
            elif "accuracy" in metric:
                temp_dict["accuracy"] = metric["accuracy"]
                temp_dict["accuracy"] = initialize_key(
                    temp_dict["accuracy"], "threshold", 0.5
                )
            elif "f1" in metric:
                temp_dict["f1"] = metric["f1"]
                temp_dict["f1"] = initialize_key(temp_dict["f1"], "average", "weighted")
                temp_dict["f1"] = initialize_key(temp_dict["f1"], "multi_class", True)
                temp_dict["f1"] = initialize_key(
                    temp_dict["f1"], "mdmc_average", "samplewise"
                )
                temp_dict["f1"] = initialize_key(temp_dict["f1"], "threshold", 0.5)
            elif "precision" in metric:
                temp_dict["precision"] = metric["precision"]
                temp_dict["precision"] = initialize_key(
                    temp_dict["precision"], "average", "weighted"
                )
                temp_dict["precision"] = initialize_key(
                    temp_dict["precision"], "multi_class", True
                )
                temp_dict["precision"] = initialize_key(
                    temp_dict["precision"], "mdmc_average", "samplewise"
                )
                temp_dict["precision"] = initialize_key(
                    temp_dict["precision"], "threshold", 0.5
                )
            elif "recall" in metric:
                temp_dict["recall"] = metric["recall"]
                temp_dict["recall"] = initialize_key(
                    temp_dict["recall"], "average", "weighted"
                )
                temp_dict["recall"] = initialize_key(
                    temp_dict["recall"], "multi_class", True
                )
                temp_dict["recall"] = initialize_key(
                    temp_dict["recall"], "mdmc_average", "samplewise"
                )
                temp_dict["recall"] = initialize_key(
                    temp_dict["recall"], "threshold", 0.5
                )
            elif "iou" in metric:
                temp_dict["iou"] = metric["iou"]
                temp_dict["iou"] = initialize_key(
                    temp_dict["iou"], "reduction", "elementwise_mean"
                )
                temp_dict["iou"] = initialize_key(temp_dict["iou"], "threshold", 0.5)

        ## need to find a better way to do this
        # special case for accuracy
        if "accuracy" in params["metrics"]:
            temp_dict["accuracy"] = initialize_key(
                temp_dict["accuracy"], "threshold", 0.5
            )

        # special case for precision
        if "precision" in params["metrics"]:
            temp_dict["precision"] = initialize_key(
                temp_dict["precision"], "average", "weighted"
            )
            temp_dict["precision"] = initialize_key(
                temp_dict["precision"], "multi_class", True
            )
            temp_dict["precision"] = initialize_key(
                temp_dict["precision"], "mdmc_average", "samplewise"
            )
            temp_dict["precision"] = initialize_key(
                temp_dict["precision"], "threshold", 0.5
            )

        # special case for f1
        if "f1" in params["metrics"]:
            temp_dict["f1"] = initialize_key(temp_dict["f1"], "average", "weighted")
            temp_dict["f1"] = initialize_key(temp_dict["f1"], "multi_class", True)
            temp_dict["f1"] = initialize_key(
                temp_dict["f1"], "mdmc_average", "samplewise"
            )
            temp_dict["f1"] = initialize_key(temp_dict["f1"], "threshold", 0.5)

        # special case for recall
        if "recall" in params["metrics"]:
            temp_dict["recall"] = initialize_key(
                temp_dict["recall"], "average", "weighted"
            )
            temp_dict["recall"] = initialize_key(
                temp_dict["recall"], "multi_class", True
            )
            temp_dict["recall"] = initialize_key(
                temp_dict["recall"], "mdmc_average", "samplewise"
            )
            temp_dict["recall"] = initialize_key(temp_dict["recall"], "threshold", 0.5)

        # special case for iou
        if "iou" in params["metrics"]:
            temp_dict["iou"] = initialize_key(
                temp_dict["iou"], "reduction", "elementwise_mean"
            )
            temp_dict["iou"] = initialize_key(temp_dict["iou"], "threshold", 0.5)

        params["metrics"] = temp_dict

    else:
        sys.exit("The key 'metrics' needs to be defined")

    # this is NOT a required parameter - a user should be able to train with NO augmentations
    params = initialize_key(params, "data_augmentation")
    if not (params["data_augmentation"] == None):
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

            # special case for random blur/noise - which takes a std-dev range
            for std_aug in ["blur", "noise"]:
                if std_aug in params["data_augmentation"]:
                    params["data_augmentation"][std_aug] = initialize_key(
                        params["data_augmentation"][std_aug], "std", [0, 1]
                    )

            # special case for random noise - which takes a mean range
            if "noise" in params["data_augmentation"]:
                params["data_augmentation"]["noise"] = initialize_key(
                    params["data_augmentation"]["noise"], "mean", 0
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
                    params["data_augmentation"], "colorjitter"
                )
                for key in ["brightness", "contrast", "saturation"]:
                    params["data_augmentation"]["colorjitter"] = initialize_key(
                        params["data_augmentation"]["colorjitter"], key, [0, 1]
                    )
                params["data_augmentation"]["colorjitter"] = initialize_key(
                    params["data_augmentation"]["colorjitter"], "hue", [-0.5, 0.5]
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

            # for all others, ensure probability is present
            if "default_probability" not in params["data_augmentation"]:
                params["data_augmentation"]["default_probability"] = 0.5

            for key in params["data_augmentation"]:
                if key != "default_probability":
                    params["data_augmentation"][key] = initialize_key(
                        params["data_augmentation"][key],
                        "probability",
                        params["data_augmentation"]["default_probability"],
                    )

    # this is NOT a required parameter - a user should be able to train with NO built-in pre-processing
    params = initialize_key(params, "data_preprocessing")
    if not (params["data_preprocessing"] == None):
        # perform this only when pre-processing is defined
        if len(params["data_preprocessing"]) > 0:
            thresholdOrClip = False
            # this can be extended, as required
            thresholdOrClipDict = [
                "threshold",
                "clip",
                "clamp",
            ]

            if (
                "resize" in params["data_preprocessing"]
                and "resample" in params["data_preprocessing"]
            ):
                print(
                    "WARNING: 'resize' is ignored as 'resample' is defined under 'data_processing'",
                    file=sys.stderr,
                )

            # iterate through all keys
            for key in params["data_preprocessing"]:  # iterate through all keys
                # for threshold or clip, ensure min and max are defined
                if not thresholdOrClip:
                    if key in thresholdOrClipDict:
                        thresholdOrClip = True  # we only allow one of threshold or clip to occur and not both
                        # initialize if nothing is present
                        if not (isinstance(params["data_preprocessing"][key], dict)):
                            params["data_preprocessing"][key] = {}

                        # if one of the required parameters is not present, initialize with lowest/highest possible values
                        # this ensures the absence of a field doesn't affect processing
                        if not "min" in params["data_preprocessing"][key]:
                            params["data_preprocessing"][key][
                                "min"
                            ] = sys.float_info.min
                        if not "max" in params["data_preprocessing"][key]:
                            params["data_preprocessing"][key][
                                "max"
                            ] = sys.float_info.max
                elif key in thresholdOrClipDict:
                    sys.exit("Use only 'threshold' or 'clip', not both")

    if "model" in params:

        if not (isinstance(params["model"], dict)):
            sys.exit("The 'model' parameter needs to be populated as a dictionary")
        elif len(params["model"]) == 0:  # only proceed if something is defined
            sys.exit(
                "The 'model' parameter needs to be populated as a dictionary and should have all properties present"
            )

        if "amp" in params["model"]:
            pass
        else:
            print("NOT using Mixed Precision Training")
            params["model"]["amp"] = False

        if "norm_type" in params["model"]:
            pass
        else:
            print("WARNING: Initializing 'norm_type' as 'batch'")
            params["model"]["norm_type"] = "batch"

        if not ("architecture" in params["model"]):
            sys.exit("The 'model' parameter needs 'architecture' key to be defined")
        if not ("final_layer" in params["model"]):
            sys.exit("The 'model' parameter needs 'final_layer' key to be defined")
        if not ("dimension" in params["model"]):
            sys.exit(
                "The 'model' parameter needs 'dimension' key to be defined, which should either 2 or 3"
            )
        if not ("base_filters" in params["model"]):
            base_filters = 32
            params["model"]["base_filters"] = base_filters
            print("Using default 'base_filters' in 'model': ", base_filters)
        if not ("class_list" in params["model"]):
            params["model"]["class_list"] = []  # ensure that this is initialized
        if not ("ignore_label_validation" in params["model"]):
            params["model"]["ignore_label_validation"] = None
        if not ("batch_norm" in params["model"]):
            params["model"]["batch_norm"] = False

        channel_keys_to_check = ["n_channels", "channels", "model_channels"]
        for key in channel_keys_to_check:
            if key in params["model"]:
                params["model"]["num_channels"] = params["model"][key]
                break

        if not ("norm_type" in params["model"]):
            print("Using default 'norm_type' in 'model': batch")
            params["model"]["norm_type"] = "batch"

    else:
        sys.exit("The 'model' parameter needs to be populated as a dictionary")

    if isinstance(params["model"]["class_list"], str):
        if ("||" in params["model"]["class_list"]) or (
            "&&" in params["model"]["class_list"]
        ):
            # special case for multi-class computation - this needs to be handled during one-hot encoding mask construction
            print(
                "WARNING: This is a special case for multi-class computation, where different labels are processed together, `reverse_one_hot` will need mapping information to work correctly"
            )
            temp_classList = params["model"]["class_list"]
            temp_classList = temp_classList.replace(
                "[", ""
            )  # we don't need the brackets
            temp_classList = temp_classList.replace(
                "]", ""
            )  # we don't need the brackets
            params["model"]["class_list"] = temp_classList.split(",")
        else:
            try:
                params["model"]["class_list"] = ast.literal_eval(
                    params["model"]["class_list"]
                )
            except AssertionError:
                AssertionError("Could not evaluate the 'class_list' in 'model'")

    if "kcross_validation" in params:
        sys.exit(
            "'kcross_validation' is no longer used, please use 'nested_training' instead"
        )

    if not ("nested_training" in params):
        sys.exit("The parameter 'nested_training' needs to be defined")
    if not ("testing" in params["nested_training"]):
        if not ("holdout" in params["nested_training"]):
            kfolds = -5
            print("Using default folds for testing split: ", kfolds)
        else:
            print(
                "WARNING: 'holdout' should not be defined under 'nested_training', please use 'testing' instead;",
                file=sys.stderr,
            )
            kfolds = params["nested_training"]["holdout"]
        params["nested_training"]["testing"] = kfolds
    if not ("validation" in params["nested_training"]):
        kfolds = -5
        print("Using default folds for validation split: ", kfolds)
        params["nested_training"]["validation"] = kfolds

    parallel_compute_command = ""
    if "parallel_compute_command" in params:
        parallel_compute_command = params["parallel_compute_command"]
        parallel_compute_command = parallel_compute_command.replace("'", "")
        parallel_compute_command = parallel_compute_command.replace('"', "")
    params["parallel_compute_command"] = parallel_compute_command

    if "opt" in params:
        DeprecationWarning("'opt' has been superceded by 'optimizer'")
        params["optimizer"] = params["opt"]

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

    if isinstance(params["optimizer"], str):
        temp_dict = {}
        temp_dict["type"] = params["optimizer"]
        params["optimizer"] = temp_dict

    return params
