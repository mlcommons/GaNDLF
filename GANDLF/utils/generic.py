import os, datetime, sys
import numpy as np
import torch
import SimpleITK as sitk
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


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

    # adding check to address https://github.com/mlcommons/GaNDLF/issues/53
    # there is quite possibly a better way to do this
    unique = np.unique(patch_size_to_check)
    if (unique.shape[0] == 1) and (unique[0] < number):
        return False
    return True


def get_date_time():
    """
    Get a well-parsed date string

    Returns:
        str: The date in format YYYY/MM/DD::HH:MM:SS
    """
    now = datetime.datetime.now().strftime("%Y/%m/%d::%H:%M:%S")
    return now


def get_unique_timestamp():
    """
    Get a well-parsed timestamp string to be used for unique filenames

    Returns:
        str: The date in format YYYYMMDD_HHMMSS
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
            "Incompatible version of GaNDLF detected ("
            + str(version_to_check_int)
            + ")"
        )

    return True


def checkPatchDimensions(patch_size, numlay):
    """
    This function checks the divisibility of a numpy array or integer for architectural integrity

    Args:
        patch_size (numpy.array): The patch size for checking.
        number (int, optional): The number to check divisibility for. Defaults to 16.

    Returns:
        int: Largest multiple of 2 (less than or equal to numlay) that each element of patch size is divisible by to yield int >= 2
    """
    if isinstance(patch_size, int):
        patch_size_to_check = np.array(patch_size)
    else:
        patch_size_to_check = patch_size
    # for 2D, don't check divisibility of last dimension
    if patch_size_to_check[-1] == 1:
        patch_size_to_check = patch_size_to_check[:-1]

    if all(
        [x >= 2 ** (numlay + 1) and x % 2**numlay == 0 for x in patch_size_to_check]
    ):
        return numlay
    else:
        # base2 = np.floor(np.log2(patch_size_to_check))
        base2 = np.array([getBase2(x) for x in patch_size_to_check])
        remain = patch_size_to_check / 2**base2  # check that at least 1

        layers = np.where(remain == 1, base2 - 1, base2)
        return int(np.min(layers))


def getBase2(num):
    # helper for checkPatchDimensions (returns the largest multiple of 2 that num is evenly divisible by)
    base = 0
    while num % 2 == 0:
        num = num / 2
        base = base + 1
    return base


def get_array_from_image_or_tensor(input_tensor_or_image):
    """
    This function returns the numpy array from a tensor or image.
    Args:
        input_tensor_or_image (torch.Tensor or sitk.Image): The input tensor or image.
    Returns:
        numpy.array: The numpy array from the tensor or image.
    """
    if isinstance(input_tensor_or_image, torch.Tensor):
        return input_tensor_or_image.detach().cpu().numpy()
    elif isinstance(input_tensor_or_image, sitk.Image):
        return sitk.GetArrayFromImage(input_tensor_or_image)
    elif isinstance(input_tensor_or_image, np.ndarray):
        return input_tensor_or_image
    else:
        raise ValueError("Input must be a torch.Tensor or sitk.Image or np.ndarray")
