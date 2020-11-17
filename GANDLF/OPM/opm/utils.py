import numpy as np
from skimage.filters.rank import maximum
from skimage.filters import gaussian
from skimage.morphology.selem import disk
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.color.colorconv import rgb2hsv
from .config import *
import matplotlib.pyplot as plt

# RGB Masking (pen) constants
RGB_RED_CHANNEL = 0
RGB_GREEN_CHANNEL = 1
RGB_BLUE_CHANNEL = 2
MIN_COLOR_DIFFERENCE = 40

# HSV Masking
HSV_HUE_CHANNEL = 0
HSV_SAT_CHANNEL = 1
HSV_VAL_CHANNEL = 2
MIN_SAT = 20 / 255
MIN_VAL = 30 / 255

# LAB Masking
LAB_L_CHANNEL = 0
LAB_A_CHANNEL = 1
LAB_B_CHANNEL = 2
LAB_L_THRESHOLD = 0.80



def pass_method(*args):
    return ""


def get_nonzero_percent(image):
    # Get first channel to simplify
    np_img = np.asarray(image)
    non_zero = np.count_nonzero(np_img)
    return non_zero / (np_img.shape[0] * np_img.shape[1])


def map_values(image, dictionary):
    template = image.copy()
    for key, value in dictionary.items():
        template[image == key] = value

    return template


def display_overlay(image, mask):
    overlay = image.copy()
    overlay[~mask] = (overlay[~mask] // 1.5).astype(np.uint8)
    plt.imshow(overlay)
    plt.show()


def hue_range_mask(image, min_hue, max_hue, sat_min=0.05):
    hsv_image = rgb2hsv(image)
    h_channel = gaussian(hsv_image[:, :, HSV_HUE_CHANNEL])
    above_min = h_channel > min_hue
    below_max = h_channel < max_hue

    s_channel = gaussian(hsv_image[:, :, HSV_SAT_CHANNEL])
    above_sat = s_channel > sat_min
    return np.logical_and(np.logical_and(above_min, below_max), above_sat)


def tissue_mask(image):
    """
    Quick and dirty hue range mask for OPM. Works well on H&E.
    """
    hue_mask = hue_range_mask(image, 0.8, 0.99)
    final_mask = remove_small_holes(hue_mask)
    return final_mask


def basic_pen_mask(image):
    """
    Mask based on RGB color channel differences. Will return true where the pixels are significantly more green
    or blue than the other channels.
    TODO: Convert to OD, threshold based on green and blue densities
    TODO: Add watershedding of pen
    :param image: RGB numpy image
    :return: image mask, True pixels are pen
    """
    green_mask = np.bitwise_and(
        image[:, :, RGB_GREEN_CHANNEL] > image[:, :, RGB_GREEN_CHANNEL],
        image[:, :, RGB_GREEN_CHANNEL] - image[:, :, RGB_GREEN_CHANNEL] > MIN_COLOR_DIFFERENCE)

    blue_mask = np.bitwise_and(
        image[:, :, RGB_BLUE_CHANNEL] > image[:, :, RGB_GREEN_CHANNEL],
        image[:, :, RGB_BLUE_CHANNEL] - image[:, :, RGB_GREEN_CHANNEL] > MIN_COLOR_DIFFERENCE)

    masked_pen = np.bitwise_or(green_mask, blue_mask)
    new_mask_image = remove_small_objects(masked_pen, PEN_SIZE_THRESHOLD)

    return maximum(np.where(new_mask_image, 1, 0), disk(PEN_MASK_EXPANSION)).astype(bool)


def basic_hsv_mask(image):
    """
    Mask based on low saturation and value (gray-black colors)
    :param image: RGB numpy image
    :return: image mask, True pixels are gray-black.
    """
    hsv_image = rgb2hsv(image)
    return np.bitwise_or(hsv_image[:, :, HSV_SAT_CHANNEL] <= MIN_SAT,
                         hsv_image[:, :, HSV_VAL_CHANNEL] <= MIN_VAL)

def hybrid_mask(image):
    return ~np.bitwise_or(basic_hsv_mask(image), basic_pen_mask(image))

def trim_mask(image, mask, background_value=0, mask_func=hybrid_mask):
    """
    Set the values of single-channel image to 0 if outside of whitespace.
    :param image: RGB numpy image
    :param mask: Mask to be trimmed
    :param background_value: Value to set in mask.
    :param mask_func: Func which takes `image` as a parameter. Returns a binary mask, `True` will be background.
    :return: `mask` with excess trimmed off
    """
    mask_copy = mask.copy()
    masked_image = mask_func(image)
    mask_copy[masked_image] = background_value
    return mask_copy


def patch_size_check(img, patch_height, patch_width):
    if img.size[0] != patch_height or img.size[1] != patch_width:
        return False
    else:
        return True


def alpha_channel_check(img, alpha_thresh=0):
    img = np.asarray(img)
    alpha_channel = img[:, :, 3]

    if np.any(alpha_channel != 255):
        return False
    else:
        return True
