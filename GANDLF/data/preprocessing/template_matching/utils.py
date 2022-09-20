""" adapted from https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/tiatoolbox/tools/stainextract.py """
import numpy as np
from skimage import exposure
import cv2


def contrast_enhancer(img, low_p=2, high_p=98):
    """
    Enhancing contrast of the input image using intensity adjustment.
    This method uses both image low and high percentiles.

    Args:
        img (:class:`numpy.ndarray`): input image used to obtain tissue mask.
            Image should be uint8.
        low_p (scalar): low percentile of image values to be saturated to 0.
        high_p (scalar): high percentile of image values to be saturated to 255.
            high_p should always be greater than low_p.

    Returns:
        img (:class:`numpy.ndarray`): Image (uint8) with contrast enhanced.

    Raises:
        AssertionError: Internal errors due to invalid img type.

    Examples:
        >>> from tiatoolbox import utils
        >>> img = utils.misc.contrast_enhancer(img, low_p=2, high_p=98)

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


def get_luminosity_tissue_mask(img, threshold):
    """Get tissue mask based on the luminosity of the input image.

    Args:
        img (:class:`numpy.ndarray`): input image used to obtain tissue mask.
        threshold (float): luminosity threshold used to determine tissue area.

    Returns:
        tissue_mask (:class:`numpy.ndarray`): binary tissue mask.

    Examples:
        >>> from tiatoolbox import utils
        >>> tissue_mask = utils.misc.get_luminosity_tissue_mask(img, threshold=0.8)

    """
    img = img.astype("uint8")  # ensure input image is uint8
    img = contrast_enhancer(img, low_p=2, high_p=98)  # Contrast  enhancement
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_lab = img_lab[:, :, 0] / 255.0  # Convert to range [0,1].
    tissue_mask = l_lab < threshold

    # check it's not empty
    assert tissue_mask.sum() != 0, "Empty tissue mask computed."

    return tissue_mask


def rgb2od(img):
    """Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).

    Args:
        img (:class:`numpy.ndarray` of type :class:`numpy.uint8`): Image RGB

    Returns:
        :class:`numpy.ndarray`: Optical density RGB image.

    Examples:
        >>> from tiatoolbox.utils import transforms, misc
        >>> rgb_img = misc.imread('path/to/image')
        >>> od_img = transforms.rgb2od(rgb_img)

    """
    mask = img == 0
    img[mask] = 1
    return np.maximum(-1 * np.log(img / 255), 1e-6)


def od2rgb(od):
    """Convert from optical density (OD_RGB) to RGB.
    RGB = 255 * exp(-1*OD_RGB)

    Args:
        od (:class:`numpy.ndarray`): Optical density RGB image

    Returns:
        numpy.ndarray: Image RGB

    Examples:
        >>> from tiatoolbox.utils import transforms, misc
        >>> rgb_img = misc.imread('path/to/image')
        >>> od_img = transforms.rgb2od(rgb_img)
        >>> rgb_img = transforms.od2rgb(od_img)

    """
    od = np.maximum(od, 1e-6)
    return (255 * np.exp(-1 * od)).astype(np.uint8)


def dl_output_for_h_and_e(dictionary):
    """Return correct value for H and E from dictionary learning output.

    Args:
        dictionary (:class:`numpy.ndarray`):
            :class:`sklearn.decomposition.DictionaryLearning` output

    Returns:
        :class:`numpy.ndarray`:
            With correct values for H and E.

    """
    return_dictionary = dictionary
    if dictionary[0, 0] < dictionary[1, 0]:
        return_dictionary = dictionary[[1, 0], :]

    return return_dictionary


def h_and_e_in_right_order(v1, v2):
    """Rearrange input vectors for H&E in correct order with H as first output.

    Args:
        v1 (:class:`numpy.ndarray`):
            Input vector for stain extraction.
        v2 (:class:`numpy.ndarray`):
            Input vector for stain extraction.

    Returns:
        :class:`numpy.ndarray`:
            Input vectors in the correct order.

    """
    return_arr = np.array([v2, v1])
    if v1[0] > v2[0]:
        return_arr = np.array([v1, v2])

    return return_arr


def vectors_in_correct_direction(e_vectors):
    """Points the eigen vectors in the right direction.

    Args:
        e_vectors (:class:`numpy.ndarray`):
            Eigen vectors.

    Returns:
        :class:`numpy.ndarray`:
            Pointing in the correct direction.

    """
    if e_vectors[0, 0] < 0:
        e_vectors[:, 0] *= -1
    if e_vectors[0, 1] < 0:
        e_vectors[:, 1] *= -1

    return e_vectors
