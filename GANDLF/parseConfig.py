import sys, yaml, pkg_resources

## dictionary to define defaults for appropriate options, which are evaluated
parameter_defaults = {
    "weighted_loss": False,  # whether weighted loss is to be used or not
    "verbose": False,  # general application verbosity
    "q_verbose": False,  # queue construction verbosity
    "medcam_enabled": False,  # interpretability via medcam
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
    "amp": False,  # automatic mixed precision
    "learning_rate": 0.001,  # default learning rate
    "clip_grad": None, # clip_gradient value
}

## dictionary to define string defaults for appropriate options
parameter_defaults_string = {
    "opt": "adam",  # the optimizer
    "patch_sampler": "uniform",  # type of sampling strategy
    "scheduler": "triangle_modified",  # the default scheduler
    "loss_function": "dc",  # default loss
}


def initialize_parameter(params, parameter_to_initialize, value=None, evaluate=True):
    """
    Initializes the specified parameter with supplied value
    """
    if parameter_to_initialize in params:
        if evaluate:
            if isinstance(params[parameter_to_initialize], str):
                params[parameter_to_initialize] = eval(params[parameter_to_initialize])
    else:
        print(
            "WARNING: Initializing '" + parameter_to_initialize + "' as " + str(value)
        )
        params[parameter_to_initialize] = value

    return params


def parse_version(version_string):
    """
    Parses version string, discards last identifier (NR/alpha/beta) and returns an integer for comparison
    """
    version_string_split = version_string.split(".")
    if len(version_string_split) > 3:
        del version_string_split[-1]
    return int("".join(version_string_split))


def initialize_key(parameters, key):
    """
    This function will initialize the key in the parameters dict to 'None' if it is absent or length is zero
    """
    if key in parameters:
        if len(parameters[key]) == 0:  # if key is present but not defined
            parameters[key] = None
    else:
        parameters[key] = None  # if key is absent

    return parameters


def parseConfig(config_file_path, version_check=True):
    """
    This function parses the configuration file and returns a dictionary of parameters
    """
    with open(config_file_path) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if version_check:  # this is only to be used for testing
        if not ("version" in params):
            sys.exit(
                "The 'version' key needs to be defined in config with 'minimum' and 'maximum' fields to determine the compatibility of configuration with code base"
            )
        else:
            gandlf_version = pkg_resources.require("GANDLF")[0].version
            gandlf_version_int = parse_version(gandlf_version)
            min = parse_version(params["version"]["minimum"])
            max = parse_version(params["version"]["maximum"])
            if (min > gandlf_version_int) or (max < gandlf_version_int):
                sys.exit(
                    "Incompatible version of GANDLF detected (" + gandlf_version + ")"
                )

    if "psize" in params:
        print(
            "WARNING: 'psize' has been deprecated in favor of 'patch_size'",
            file=sys.stderr,
        )
        if not ("patch_size" in params):
            params["patch_size"] = params["psize"]

    if "patch_size" in params:
        if len(params["patch_size"]) == 2:  # 2d check
            params["patch_size"].append(
                1
            )  # ensuring same size during torchio processing
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
                        params[
                            "loss_function"
                        ] = key  # use simple string for other functions - can be extended with parameters, if needed
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

        # initialize metrics dict
        for metric in params["metrics"]:
            temp_dict[metric] = None

        # special case for accuracy
        if "accuracy" in params["metrics"]:
            if isinstance(params["metrics"], list):
                temp_dict["accuracy"] = {}
            else:
                temp_dict["accuracy"] = params["metrics"]["accuracy"]

            # accuracy needs an associated threshold, if not defined, default to '0.5'
            initialize_threshold = False
            if isinstance(temp_dict["accuracy"], dict):
                if "threshold" in temp_dict["accuracy"]:
                    pass
                else:
                    initialize_threshold = True
            else:
                initialize_threshold = True

            if initialize_threshold:
                temp_dict["accuracy"]["threshold"] = 0.5
        params["metrics"] = temp_dict

    else:
        sys.exit("The key 'metrics' needs to be defined")

    # this is NOT a required parameter - a user should be able to train with NO augmentations
    params = initialize_key(params, "data_augmentation")
    if not (params["data_augmentation"] == None):
        if len(params["data_augmentation"]) > 0:  # only when augmentations are defined

            # special case for spatial augmentation, which is now deprecated
            if "spatial" in params["data_augmentation"]:
                if not ("affine" in params["data_augmentation"]) or not (
                    "elastic" in params["data_augmentation"]
                ):
                    print(
                        "WARNING: 'spatial' is now deprecated in favor of split 'affine' and/or 'elastic'",
                        file=sys.stderr,
                    )
                    params["data_augmentation"]["affine"] = {}
                    params["data_augmentation"]["elastic"] = {}
                    del params["data_augmentation"]["spatial"]

            # special case for random swapping - which takes a patch size to swap pixels around
            if "swap" in params["data_augmentation"]:
                if not (isinstance(params["data_augmentation"]["swap"], dict)):
                    params["data_augmentation"]["swap"] = {}
                if not ("patch_size" in params["data_augmentation"]["swap"]):
                    params["data_augmentation"]["swap"]["patch_size"] = 15  # default

            # special case for random blur/noise - which takes a std-dev range
            for std_aug in ["blur", "noise"]:
                if std_aug in params["data_augmentation"]:
                    if not (isinstance(params["data_augmentation"][std_aug], dict)):
                        params["data_augmentation"][std_aug] = {}
                    if not ("std" in params["data_augmentation"][std_aug]):
                        params["data_augmentation"][std_aug]["std"] = [0, 1]  # default

            # special case for random noise - which takes a mean range
            if "noise" in params["data_augmentation"]:
                if not (isinstance(params["data_augmentation"]["noise"], dict)):
                    params["data_augmentation"]["noise"] = {}
                if not ("mean" in params["data_augmentation"]["noise"]):
                    params["data_augmentation"]["noise"]["mean"] = 0  # default

            # special case for augmentations that need axis defined
            for axis_aug in ["flip", "anisotropic"]:
                if axis_aug in params["data_augmentation"]:
                    if not (isinstance(params["data_augmentation"][axis_aug], dict)):
                        params["data_augmentation"][axis_aug] = {}
                    if not ("axis" in params["data_augmentation"][axis_aug]):
                        params["data_augmentation"][axis_aug]["axis"] = [
                            0,
                            1,
                            2,
                        ]  # default

            # special case for augmentations that need axis defined in 1,2,3
            for axis_aug in ["rotate_90", "rotate_180"]:
                if axis_aug in params["data_augmentation"]:
                    if not (isinstance(params["data_augmentation"][axis_aug], dict)):
                        params["data_augmentation"][axis_aug] = {}
                    if not ("axis" in params["data_augmentation"][axis_aug]):
                        params["data_augmentation"][axis_aug]["axis"] = [
                            1,
                            2,
                            3,
                        ]  # default

            if (
                "anisotropic" in params["data_augmentation"]
            ):  # special case for anisotropic
                if not ("downsampling" in params["data_augmentation"]["anisotropic"]):
                    default_downsampling = 1.5
                else:
                    default_downsampling = params["data_augmentation"]["anisotropic"][
                        "downsampling"
                    ]

                initialize_downsampling = False
                if type(default_downsampling) is list:
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
                        default_downsampling = 1.5
                    params["data_augmentation"]["anisotropic"][
                        "downsampling"
                    ] = default_downsampling  # default

            # for all others, ensure probability is present
            default_probability = 0.5
            if "default_probability" in params["data_augmentation"]:
                default_probability = float(
                    params["data_augmentation"]["default_probability"]
                )
            for key in params["data_augmentation"]:
                if key != "default_probability":
                    if (params["data_augmentation"][key] == None) or not (
                        "probability" in params["data_augmentation"][key]
                    ):  # when probability is not present for an augmentation, default to '1'
                        if not isinstance(params["data_augmentation"][key], dict):
                            params["data_augmentation"][key] = {}
                        params["data_augmentation"][key][
                            "probability"
                        ] = default_probability

    # this is NOT a required parameter - a user should be able to train with NO built-in pre-processing
    params = initialize_key(params, "data_preprocessing")
    if not (params["data_preprocessing"] == None):
        if (
            len(params["data_preprocessing"]) < 0
        ):  # perform this only when pre-processing is defined
            thresholdOrClip = False
            thresholdOrClipDict = [
                "threshold",
                "clip",
            ]  # this can be extended, as required
            keysForWarning = [
                "resize"
            ]  # properties for which the user will see a warning

            # iterate through all keys
            for key in params["data_preprocessing"]:  # iterate through all keys
                # for threshold or clip, ensure min and max are defined
                if not thresholdOrClip:
                    if key in thresholdOrClipDict:
                        thresholdOrClip = True  # we only allow one of threshold or clip to occur and not both
                        if not (
                            isinstance(params["data_preprocessing"][key], dict)
                        ):  # initialize if nothing is present
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
                else:
                    sys.exit("Use only 'threshold' or 'clip', not both")

                # give a warning for resize
                if key in keysForWarning:
                    print(
                        "WARNING: '"
                        + key
                        + "' is generally not recommended, as it changes image properties in unexpected ways.",
                        file=sys.stderr,
                    )

    if "modelName" in params:
        defineDefaultModel = False
        print("This option has been superceded by 'model'", file=sys.stderr)
        which_model = str(params["modelName"])
    elif "which_model" in params:
        defineDefaultModel = False
        print("This option has been superceded by 'model'", file=sys.stderr)
        which_model = str(params["which_model"])
    else:  # default case
        defineDefaultModel = True
    if defineDefaultModel == True:
        which_model = "resunet"
        # print('Using default model: ', which_model)
    params["which_model"] = which_model

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

    else:
        sys.exit("The 'model' parameter needs to be populated as a dictionary")

    if isinstance(params["model"]["class_list"], str):
        try:
            params["model"]["class_list"] = eval(params["model"]["class_list"])
        except:
            if ("||" in params["model"]["class_list"]) or (
                "&&" in params["model"]["class_list"]
            ):
                # special case for multi-class computation - this needs to be handled during one-hot encoding mask construction
                print(
                    "This is a special case for multi-class computation, where different labels are processed together"
                )
                temp_classList = params["model"]["class_list"]
                temp_classList = temp_classList.replace(
                    "[", ""
                )  # we don't need the brackets
                temp_classList = temp_classList.replace(
                    "]", ""
                )  # we don't need the brackets
                params["model"]["class_list"] = temp_classList.split(",")

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

    return params
