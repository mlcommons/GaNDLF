import os, datetime, sys
import numpy as np

from GANDLF.models import get_model
from GANDLF.schedulers import get_scheduler
from GANDLF.optimizers import get_optimizer
from GANDLF.data import (
    get_train_loader,
    get_validation_loader,
    get_penalty_loader,
    ImagesFromDataFrame,
)
from GANDLF.utils import (
    populate_channel_keys_in_params,
    populate_header_in_parameters,
    parseTrainingCSV,
    send_model_to_device,
)

from .tensor import get_class_imbalance_weights


def checkPatchDivisibility(patch_size, number=16):
    """
    This function checks the divisibility of a numpy array or integer for architectural integrity

    Args:
        patch_size (numpy.array): The patch size for checking.
        number (int, optional): The number to check divisibility for. Defaults to 16.

    Returns:
        bool: If all elements of array are divisible or not, after taking 2D patches into account.
    """
    if isinstance(patch_size, int):
        patch_size_to_check = np.array(patch_size)
    else:
        patch_size_to_check = patch_size
    # for 2D, don't check divisibility of last dimension
    if patch_size_to_check[-1] == 1:
        patch_size_to_check = patch_size_to_check[:-1]
    # for 2D, don't check divisibility of first dimension
    elif patch_size_to_check[0] == 1:
        patch_size_to_check = patch_size_to_check[1:]
    if np.count_nonzero(np.remainder(patch_size_to_check, number)) > 0:
        return False

    # adding check to address https://github.com/CBICA/GaNDLF/issues/53
    # there is quite possibly a better way to do this
    unique = np.unique(patch_size_to_check)
    if (unique.shape[0] == 1) and (unique[0] <= number):
        return False
    return True


def fix_paths(cwd):
    """
    This function takes the current working directory of the script (which is required for VIPS) and sets up all the paths correctly

    Args:
        cwd (str): The current working directory.
    """
    if os.name == "nt":  # proceed for windows
        vipshome = os.path.join(cwd, "vips/vips-dev-8.10/bin")
        os.environ["PATH"] = vipshome + ";" + os.environ["PATH"]


def get_date_time():
    """
    Get a well-parsed date string

    Returns:
        str: The date in format YYYY/MM/DD::HH:MM:SS
    """
    now = datetime.datetime.now().strftime("%Y/%m/%d::%H:%M:%S")
    return now


def get_filename_extension_sanitized(filename):
    """
    This function returns the extension of the filename with leading and trailing characters removed.
    Args:
        filename (str): The filename to be processed.
    Returns:
        str: The filename with extension removed.
    """
    _, ext = os.path.splitext(filename)
    # if .gz or .nii file is detected, always return .nii.gz
    if (ext == ".gz") or (ext == ".nii"):
        ext = ".nii.gz"
    return ext


def parse_version(version_string):
    """
    Parses version string, discards last identifier (NR/alpha/beta) and returns an integer for comparison.

    Args:
        version_string (str): The string to be parsed.

    Returns:
        int: The version number.
    """
    version_string_split = version_string.split(".")
    if len(version_string_split) > 3:
        del version_string_split[-1]
    return int("".join(version_string_split))


def version_check(version_from_config, version_to_check):
    """
    This function checks if the version of the config file is compatible with the version of the code.

    Args:
        version_from_config (str): The version of the config file.
        version_to_check (str): The version of the code or model to check.

    Returns:
        bool: If the version of the config file is compatible with the version of the code.
    """
    version_to_check_int = parse_version(version_to_check)
    min_ver = parse_version(version_from_config["minimum"])
    max_ver = parse_version(version_from_config["maximum"])
    if (min_ver > version_to_check_int) or (max_ver < version_to_check_int):
        sys.exit(
            "Incompatible version of GaNDLF detected (" + version_to_check_int + ")"
        )

    return True


def create_pytorch_objects(parameters, train_csv, val_csv, device):
    """
    _summary_

    Args:
        parameters (_type_): _description_
        train_csv (_type_): _description_
        val_csv (_type_): _description_
        device (_type_): _description_

    Returns:
        model (_type_): _description_
        optimizer (_type_): _description_
        train_loader (_type_): _description_
        val_loader (_type_): _description_
        scheduler (_type_): _description_
    """
    # populate the data frames
    parameters["training_data"], headers_train = parseTrainingCSV(train_csv, train=True)
    parameters = populate_header_in_parameters(parameters, headers_train)
    parameters["validation_data"], _ = parseTrainingCSV(val_csv, train=False)
    
    # get the model
    model = get_model(parameters)
    parameters["model_parameters"] = model.parameters()

    # send model to correct device
    model, parameters["model"]["amp"], parameters["device"] = send_model_to_device(
        model, amp=parameters["model"]["amp"], device=device, optimizer=optimizer
    )

    # get the optimizer
    optimizer = get_optimizer(parameters, model)
    parameters["optimizer_object"] = optimizer
    # get the train loader
    train_loader = get_train_loader(parameters)
    # get the validation loader
    val_loader = get_validation_loader(parameters)

    
    if not ("step_size" in parameters["scheduler"]):
        parameters["scheduler"]["step_size"] = (
            parameters["training_samples_size"] / parameters["learning_rate"]
        )

    scheduler = get_scheduler(parameters)

    # these keys contain generators, and are not needed beyond this point in params
    generator_keys_to_remove = ["optimizer_object", "model_parameters"]
    for key in generator_keys_to_remove:
        parameters.pop(key, None)

    validation_data_for_torch = ImagesFromDataFrame(
        parameters["validation_data"], parameters, train=False, loader_type="validation"
    )
    # Fetch the appropriate channel keys
    # Getting the channels for training and removing all the non numeric entries from the channels
    parameters = populate_channel_keys_in_params(validation_data_for_torch, parameters)
    
    # Calculate the weights here
    if parameters["weighted_loss"]:
        print("Calculating weights for loss")
        penalty_loader = get_penalty_loader(parameters)

        parameters["weights"], parameters["class_weights"] = get_class_imbalance_weights(
            penalty_loader, parameters
        )
        del penalty_loader
    else:
        parameters["weights"], parameters["class_weights"] = None, None
    
    return model, optimizer, train_loader, val_loader, scheduler, parameters
