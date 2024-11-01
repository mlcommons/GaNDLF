import sys, os
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import skimage.io

# from skimage.filters.rank import maximum
from skimage.filters import gaussian

# from skimage.morphology.footprints import disk
from skimage.morphology import remove_small_holes
from skimage.color.colorconv import rgb2hsv
import cv2

# from skimage.exposure import rescale_intensity
# from skimage.color import rgb2hed

# import matplotlib.pyplot as plt
import yaml
import openslide

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


def print_sorted_dict(dictionary: dict) -> str:
    """
    Print a dictionary with sorted keys.

    Args:
        dictionary (dict): The input dictionary.

    Returns:
        str: The sorted dictionary.
    """
    sorted_keys = sorted(list(dictionary.keys()))
    output_str = "{"
    for index, key in enumerate(sorted_keys):
        output_str += str(key) + ": " + str(dictionary[key])
        if index < len(sorted_keys) - 1:
            output_str += "; "
    output_str += "}"

    return output_str


def convert_to_tiff(
    filename: str,
    output_dir: str,
    updated_file_name_identifier: Optional[str] = "converted",
) -> str:
    """
    Convert an image to tiff.

    Args:
        filename (str): The input filename.
        output_dir (str): The output directory.
        updated_file_name_identifier (str, optional): The identifier to use for the updated file name. Defaults to "converted".

    Returns:
        str: The path to the converted image.
    """
    base, ext = os.path.splitext(filename)
    # for png or jpg images, write image back to tiff
    if ext in [".png", ".jpg", ".jpeg"]:
        converted_img_path = os.path.join(output_dir, "tiff_converted")
        Path(converted_img_path).mkdir(parents=True, exist_ok=True)
        temp_file = os.path.join(
            converted_img_path,
            os.path.basename(base) + "_" + updated_file_name_identifier + ".tiff",
        )
        temp_img = skimage.io.imread(filename)
        skimage.io.imsave(temp_file, temp_img)
        return temp_file
    else:
        return filename


def pass_method(*args: object) -> str:
    return ""


def get_nonzero_percent(image: np.ndarray) -> float:
    """
    Get the percentage of non-zero pixels in an image.

    Args:
        image (np.ndarray): The input image.

    Returns:
        float: The percentage of non-zero pixels.
    """
    np_img = np.asarray(image)
    non_zero = np.count_nonzero(np_img)
    return non_zero / (np_img.shape[0] * np_img.shape[1])


def get_patch_class_proportions(image: np.ndarray) -> dict:
    """
    Get the class proportions of a patch.

    Args:
        image (np.ndarray): The input image.

    Returns:
        dict: The class proportions
    """
    np_img = np.asarray(image)
    unique, counts = np.unique(image, return_counts=True)
    denom = np_img.shape[0] * np_img.shape[1]
    prop_dict = {val: count / denom for val, count in list(zip(unique, counts))}
    return print_sorted_dict(prop_dict)


def map_values(image: np.ndarray, dictionary: dict) -> np.ndarray:
    """
    Map values in an image to a new set of values.

    Args:
        image (np.ndarray): The input image.
        dictionary (dict): The dictionary to use for mapping.

    Returns:
        np.ndarray: The mapped image.
    """
    template = image.copy()  # Copy image so all values not in dict are unmodified
    for key, value in dictionary.items():
        template[image == key] = value

    return template


## commented because this is GUI and is not used in the code
# def display_overlay(image, mask):
#     overlay = image.copy()
#     overlay[~mask] = (overlay[~mask] // 1.5).astype(np.uint8)
#     plt.imshow(overlay)
#     plt.show()


def hue_range_mask(
    image: np.ndarray, min_hue: float, max_hue: float, sat_min: Optional[float] = 0.05
) -> np.ndarray:
    """
    Mask based on hue range.

    Args:
        image (np.ndarray): RGB numpy image
        min_hue (float): Minimum hue value
        max_hue (float): Maximum hue value
        sat_min (Optional[float], optional): Minimum saturation value. Defaults to 0.05.

    Returns:
        np.ndarray: image mask, True pixels are within the hue range.
    """
    hsv_image = rgb2hsv(image)
    h_channel = gaussian(hsv_image[:, :, HSV_HUE_CHANNEL])
    above_min = h_channel > min_hue
    below_max = h_channel < max_hue

    s_channel = gaussian(hsv_image[:, :, HSV_SAT_CHANNEL])
    above_sat = s_channel > sat_min
    return np.logical_and(np.logical_and(above_min, below_max), above_sat)


def tissue_mask(image: np.ndarray) -> np.ndarray:
    """
    Mask based on low saturation and value (gray-black colors)

    Args:
        image (np.ndarray): RGB numpy image

    Returns:
        np.ndarray: image mask, True pixels are gray-black.
    """
    hue_mask = hue_range_mask(image, 0.8, 0.99)
    final_mask = remove_small_holes(hue_mask)
    return final_mask


### unused function because pen_size_threshold and pen_mask_expansion are not defined
# def basic_pen_mask(image, pen_size_threshold, pen_mask_expansion):
#     green_mask = np.bitwise_and(
#         image[:, :, RGB_GREEN_CHANNEL] > image[:, :, RGB_GREEN_CHANNEL],
#         image[:, :, RGB_GREEN_CHANNEL] - image[:, :, RGB_GREEN_CHANNEL]
#         > MIN_COLOR_DIFFERENCE,
#     )

#     blue_mask = np.bitwise_and(
#         image[:, :, RGB_BLUE_CHANNEL] > image[:, :, RGB_GREEN_CHANNEL],
#         image[:, :, RGB_BLUE_CHANNEL] - image[:, :, RGB_GREEN_CHANNEL]
#         > MIN_COLOR_DIFFERENCE,
#     )

#     masked_pen = np.bitwise_or(green_mask, blue_mask)
#     new_mask_image = remove_small_objects(masked_pen, pen_size_threshold)

#     return maximum(np.where(new_mask_image, 1, 0), disk(pen_mask_expansion)).astype(
#         bool
#     )


### unused function
# def basic_hsv_mask(image):
#     """
#     Mask based on low saturation and value (gray-black colors)
#     :param image: RGB numpy image
#     :return: image mask, True pixels are gray-black.
#     """
#     hsv_image = rgb2hsv(image)
#     return np.bitwise_or(
#         hsv_image[:, :, HSV_SAT_CHANNEL] <= MIN_SAT,
#         hsv_image[:, :, HSV_VAL_CHANNEL] <= MIN_VAL,
#     )


### unused function because pen_size_threshold and pen_mask_expansion are not defined
# def hybrid_mask(image, pen_size_threshold, pen_mask_expansion):
#     return ~np.bitwise_or(basic_hsv_mask(image), basic_pen_mask(image, pen_size_threshold, pen_mask_expansion))


### unused function because pen_size_threshold and pen_mask_expansion are not defined
# def trim_mask(image, mask, pen_size_threshold, pen_mask_expansion, background_value=0, mask_func=hybrid_mask):
#     """
#     Set the values of single-channel image to 0 if outside of whitespace.
#     :param image: RGB numpy image
#     :param mask: Mask to be trimmed
#     :param background_value: Value to set in mask.
#     :param mask_func: Func which takes `image` as a parameter. Returns a binary mask, `True` will be background.
#     :return: `mask` with excess trimmed off
#     """
#     mask_copy = mask.copy()
#     masked_image = mask_func(image)
#     mask_copy[masked_image] = background_value
#     return mask_copy


def patch_size_check(img: np.ndarray, patch_height: int, patch_width: int) -> bool:
    """
    This function checks if the patch size is valid.

    Args:
        img (np.ndarray): Input image.
        patch_height (int): The height of the patch.
        patch_width (int): The width of the patch.

    Returns:
        bool: Whether or not the patch size is valid.
    """
    img = np.asarray(img)

    return_val = False
    if not (img.shape[0] != patch_height or img.shape[1] != patch_width):
        return_val = True

    return return_val


def alpha_rgb_2d_channel_check(img: np.ndarray) -> bool:
    """
    This function checks if an image has a valid alpha channel.

    Args:
        img (np.ndarray): Input image.

    Returns:
        bool: Whether or not the image has alpha channel.
    """
    img = np.asarray(img)
    # If the image has three dimensions AND there is no alpha_channel...
    if len(img.shape) == 3 and img.shape[-1] == 3:
        return True
    # If the image has three dimensions AND there IS an alpha channel...
    elif len(img.shape) == 3 and img.shape[-1] == 4:
        alpha_channel = img[:, :, 3]

        if np.any(alpha_channel != 255):
            return False
        else:
            return True
    # If the image is two dims, return True
    elif len(img.shape) == 2:
        return True
    # Other images (4D, RGBA+____, etc.), return False.
    else:
        return False


# def pen_marking_check(img, intensity_thresh=225, intensity_thresh_saturation =50, intensity_thresh_b = 128):
#    """
#    This function is used to curate patches from the input image. It is used to remove patches that have pen markings.
#    Args:
#        img (np.ndarray): Input Patch Array to check the artifact/background.
#        intensity_thresh (int, optional): Threshold to check whiteness in the patch. Defaults to 225.
#        intensity_thresh_saturation (int, optional): Threshold to check saturation in the patch. Defaults to 50.
#        intensity_thresh_b (int, optional) : Threshold to check blackness in the patch
#        patch_size (int, optional): Tiling Size of the WSI/patch size. Defaults to 256. patch_size=config["patch_size"]
#    Returns:
#        bool: Whether the patch is valid (True) or not (False)
#    """
#    patch_size= (256,256)
#    ihc_hed = rgb2hed(img)
#    patch_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#    e = rescale_intensity(ihc_hed[:, :, 1], out_range=(0, 255), in_range=(0, np.percentile(ihc_hed[:, :, 1], 99)))
#    if np.sum(e < 50) / (patch_size[0] * patch_size[1]) > 0.95 or (np.sum(patch_hsv[...,0] < 128) / (patch_size[0] * patch_size[1])) > 0.97:
#        return False
#    #Assume patch is valid
#    return True


def patch_artifact_check(
    img: np.ndarray,
    intensity_thresh: int = 250,
    intensity_thresh_saturation: int = 5,
    intensity_thresh_b: int = 128,
    patch_size: Optional[List[int]] = [256, 256],
) -> bool:
    """
    This function is used to curate patches from the input image. It is used to remove patches that have artifacts.

    Args:
        img (np.ndarray): Input Patch Array to check the artifact/background.
        intensity_thresh (int, optional): Threshold to check whiteness in the patch. Defaults to 250.
        intensity_thresh_saturation (int, optional): Threshold to check saturation in the patch. Defaults to 5.
        intensity_thresh_b (int, optional): Threshold to check blackness in the patch. Defaults to 128.
        patch_size (Optional[List[int]], optional): Tiling Size of the WSI/patch size. Defaults to [256, 256].

    Returns:
        bool: Whether the patch is valid or not.
    """
    # patch_size = config["patch_size"]
    patch_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    count_white_pixels = np.sum(np.logical_and.reduce(img > intensity_thresh, axis=2))
    percent_pixels = count_white_pixels / (patch_size[0] * patch_size[1])
    count_black_pixels = np.sum(np.logical_and.reduce(img < intensity_thresh_b, axis=2))
    percent_pixel_b = count_black_pixels / (patch_size[0] * patch_size[1])
    percent_pixel_2 = np.sum(patch_hsv[..., 1] < intensity_thresh_saturation) / (
        patch_size[0] * patch_size[1]
    )
    percent_pixel_3 = np.sum(patch_hsv[..., 2] > intensity_thresh) / (
        patch_size[0] * patch_size[1]
    )

    if (
        percent_pixel_2 > 0.99
        or np.mean(patch_hsv[..., 1]) < 5
        or percent_pixel_3 > 0.99
    ):
        if percent_pixel_2 < 0.1:
            return False
    elif (
        (percent_pixel_2 > 0.99 and percent_pixel_3 > 0.99)
        or percent_pixel_b > 0.99
        or percent_pixels > 0.9
    ):
        return False
    # assume that the patch is valid
    return True


def parse_config(config_file: str) -> dict:
    """
    Function that parses the config file.

    Args:
        config_file (str): The path to the config file.

    Returns:
        dict: The parsed config file.
    """
    config = yaml.safe_load(open(config_file, "r"))

    # initialize defaults if not specified
    config["scale"] = config.get("scale", 16)
    config["num_patches"] = config.get("num_patches", -1)
    config["num_workers"] = config.get("num_workers", 1)
    config["save_patches"] = config.get("save_patches", True)
    config["value_map"] = config.get("value_map", None)
    config["read_type"] = config.get("read_type", "random")
    config["overlap_factor"] = config.get("overlap_factor", 0.0)
    config["patch_size"] = config.get("patch_size", [256, 256])

    return config


def is_mask_too_big(mask: np.ndarray) -> bool:
    """
    Function that returns a boolean value indicating whether the mask is too big to make processing slow.

    Args:
        mask (np.ndarray): The valid mask.

    Returns:
        bool: True if the mask is too big, False otherwise.
    """

    if sys.getsizeof(mask) > 16 * (1024**2):
        return True
    else:
        return False


def generate_initial_mask(slide_path: str, scale: int) -> Tuple[np.ndarray, tuple]:
    """
    Function that generates the initial mask for the slide.

    Args:
        slide_path (str): The path to the slide.
        scale (int): The scale to use for the mask.

    Returns:
        Tuple[np.ndarray, tuple]: The valid mask and the real scale.
    """
    # Open slide and get properties
    slide = openslide.open_slide(slide_path)
    slide_dims = slide.dimensions

    # Call thumbnail for effiency, calculate scale relative to whole slide
    slide_thumbnail = np.asarray(
        slide.get_thumbnail((slide_dims[0] // scale, slide_dims[1] // scale))
    )
    real_scale = (
        slide_dims[0] / slide_thumbnail.shape[1],
        slide_dims[1] / slide_thumbnail.shape[0],
    )

    valid_mask = tissue_mask(slide_thumbnail)

    if is_mask_too_big(valid_mask):
        print(
            "Calculated tissue mask is too big; considering increasing the scale for faster processing."
        )

    return valid_mask, real_scale


def get_patch_size_in_microns(
    input_slide_path: str, patch_size_from_config: str, verbose: Optional[bool] = False
) -> List[int]:
    """
    Function that returns the patch size in pixels.

    Args:
        input_slide_path (str): The path to the slide.
        patch_size_from_config (str): The patch size from the config file.
        verbose (Optional[bool], optional): Whether to print verbose output. Defaults to False.

    Returns:
        List[int]: The patch size after getting converted to pixels.
    """
    return_patch_size = [0, 0]
    patch_size = None

    assert isinstance(
        patch_size_from_config, (str, list, tuple)
    ), "Patch size must be a list or string."

    if isinstance(patch_size_from_config, str):
        # first remove all spaces and square brackets
        patch_size_from_config = patch_size_from_config.replace(" ", "")
        patch_size_from_config = patch_size_from_config.replace("[", "")
        patch_size_from_config = patch_size_from_config.replace("]", "")
        # try different split strategies
        patch_size = patch_size_from_config.split(",")
        if len(patch_size) == 1:
            patch_size = patch_size_from_config.split("x")
        if len(patch_size) == 1:
            patch_size = patch_size_from_config.split("X")
        if len(patch_size) == 1:
            patch_size = patch_size_from_config.split("*")
        assert (
            len(patch_size) == 2
        ), "Could not parse patch size from config.yml, use either ',', 'x', 'X', or '*' as separator between x and y dimensions."
    elif isinstance(patch_size_from_config, list) or isinstance(
        patch_size_from_config, tuple
    ):
        patch_size = patch_size_from_config

    magnification_prev = -1
    for i, _ in enumerate(patch_size):
        # for i in range(len(patch_size)):
        magnification = -1
        if str(patch_size[i]).isnumeric():
            return_patch_size[i] = int(patch_size[i])
        elif isinstance(patch_size[i], str):
            if ("m" in patch_size[i]) or ("mu" in patch_size[i]):
                if verbose:
                    print(
                        "Using mpp to calculate patch size for dimension {}".format(i)
                    )
                # only enter if "m" is present in patch size
                input_slide = openslide.open_slide(input_slide_path)
                metadata = input_slide.properties
                if i == 0:
                    for _property in [
                        openslide.PROPERTY_NAME_MPP_X,
                        "tiff.XResolution",
                        "XResolution",
                    ]:
                        if _property in metadata:
                            magnification = float(metadata[_property])
                            magnification_prev = magnification
                            break
                elif i == 1:
                    for _property in [
                        openslide.PROPERTY_NAME_MPP_Y,
                        "tiff.YResolution",
                        "YResolution",
                    ]:
                        if _property in metadata:
                            magnification = float(metadata[_property])
                            break
                    if magnification == -1:
                        # if y-axis data is missing, use x-axis data
                        magnification = magnification_prev
                # get patch size in pixels
                # check for 'mu' first
                size_in_microns = patch_size[i].replace("mu", "")
                size_in_microns = float(size_in_microns.replace("m", ""))
                if verbose:
                    print(
                        "Original patch size in microns for dimension {}",
                        format(size_in_microns),
                    )
                if magnification > 0:
                    return_patch_size[i] = round(size_in_microns / magnification)
                    magnification_prev = magnification
            else:
                return_patch_size[i] = float(patch_size[i])

    if verbose:
        print(
            "Estimated patch size in pixels: [{},{}]".format(
                return_patch_size[0], return_patch_size[1]
            )
        )

    return return_patch_size
