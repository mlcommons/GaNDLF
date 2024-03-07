""" adapted from https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/tiatoolbox/tools/stainextract.py """

from typing import Optional
import numpy as np
from skimage import exposure
import cv2


def contrast_enhancer(
    img: np.ndarray, low_p: Optional[int] = 2, high_p: Optional[int] = 98
) -> np.ndarray:
    """
    Enhance contrast of an image using percentile rescaling.

    Args:
        img (np.ndarray): The input image.
        low_p (Optional[int], optional): The low percentile. Defaults to 2.
        high_p (Optional[int], optional): The high percentile. Defaults to 98.

    Returns:
        np.ndarray: The contrast enhanced image.
    """
    # check if image is not uint8
    assert img.dtype == np.uint8, "Image should be uint8"
    img_out = img.copy()
    p_low, p_high = np.percentile(img_out, (low_p, high_p))
    if p_low >= p_high:
        p_low, p_high = np.min(img_out), np.max(img_out)
    if p_high > p_low:
        img_out = exposure.rescale_intensity(
            img_out, in_range=(p_low, p_high), out_range=(0.0, 255.0)
        )
    return np.uint8(img_out)


def get_luminosity_tissue_mask(img: np.ndarray, threshold: float) -> np.ndarray:
    """
    Compute tissue mask based on luminosity thresholding.

    Args:
        img (np.ndarray): The input image.
        threshold (float): The threshold for luminosity.

    Returns:
        np.ndarray: The tissue mask.
    """
    img = img.astype("uint8")  # ensure input image is uint8
    img = contrast_enhancer(img, low_p=2, high_p=98)  # Contrast  enhancement
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_lab = img_lab[:, :, 0] / 255.0  # Convert to range [0,1].
    tissue_mask = l_lab < threshold

    # check it's not empty
    assert tissue_mask.sum() != 0, "Empty tissue mask computed."

    return tissue_mask


def rgb2od(img: np.ndarray) -> np.ndarray:
    """
    Convert from RGB to optical density (OD_RGB).

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The optical density RGB image.
    """
    mask = img == 0
    img[mask] = 1
    return np.maximum(-1 * np.log(img / 255), 1e-6)


def od2rgb(od: np.ndarray) -> np.ndarray:
    """
    Convert from optical density to RGB.

    Args:
        od (np.ndarray): The optical density image.

    Returns:
        np.ndarray: The RGB image.
    """
    od = np.maximum(od, 1e-6)
    return (255 * np.exp(-1 * od)).astype(np.uint8)


def dl_output_for_h_and_e(dictionary: np.ndarray) -> np.ndarray:
    """
    Rearrange dictionary for H&E in correct order with H as first output.

    Args:
        dictionary (np.ndarray): The input dictionary.

    Returns:
        np.ndarray: The dictionary in the correct order.
    """
    return_dictionary = dictionary
    if dictionary[0, 0] < dictionary[1, 0]:
        return_dictionary = dictionary[[1, 0], :]

    return return_dictionary


def h_and_e_in_right_order(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Rearrange vectors for H&E in correct order with H as first output.

    Args:
        v1 (np.ndarray): The first vector for stain extraction.
        v2 (np.ndarray): The second vector for stain extraction.

    Returns:
        np.ndarray: The vectors in the correct order.
    """
    return_arr = np.array([v2, v1])
    if v1[0] > v2[0]:
        return_arr = np.array([v1, v2])

    return return_arr


def vectors_in_correct_direction(e_vectors: np.ndarray) -> np.ndarray:
    """
    Ensure that the vectors are in the correct direction.

    Args:
        e_vectors (np.ndarray): The input vectors.

    Returns:
        np.ndarray: The vectors in the correct direction.
    """
    if e_vectors[0, 0] < 0:
        e_vectors[:, 0] *= -1
    if e_vectors[0, 1] < 0:
        e_vectors[:, 1] *= -1

    return e_vectors
