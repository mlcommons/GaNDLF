"""
All the segmentation metrics are to be called from here
"""
import sys
import torch
import numpy as np
from GANDLF.losses.segmentation import dice
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import (
    distance_transform_edt,
    binary_erosion,
    generate_binary_structure,
)


def _convert_tensor_to_int_label_array(input_tensor):
    """
    This function converts a tensor of labels to a numpy array of labels.

    Args:
        input_tensor (torch.Tensor): Input data containing objects. Can be any type but will be converted into binary: background where 0, object everywhere else.

    Returns:
        numpy.ndarray: The numpy array of labels.
    """
    result_array = input_tensor.detach().cpu().numpy()
    if result_array.shape[-1] == 1:
        result_array = result_array.squeeze(-1)
    # ensure that we are dealing with a binary array
    result_array[result_array < 0.5] = 0
    result_array[result_array >= 0.5] = 1
    return result_array.astype(np.int64)


def multi_class_dice(output, label, params, per_label=False):
    """
    This function computes a multi-class dice.

    Args:
        output (torch.Tensor): Input data containing objects. Can be any type but will be converted into binary: background where 0, object everywhere else.
        label (torch.Tensor): Input data containing objects. Can be any type but will be converted into binary: background where 0, object everywhere else.
        params (dict): The parameter dictionary containing training and data information.
        per_label (bool, optional): Whether the dice needs to be calculated per label or not. Defaults to False.

    Returns:
        float or list: The average dice for all labels or a list of per-label dice scores.
    """
    total_dice = 0
    avg_counter = 0
    per_label_dice = []
    for i in range(0, params["model"]["num_classes"]):
        # this check should only happen during validation
        if i != params["model"]["ignore_label_validation"]:
            current_dice = dice(output[:, i, ...], label[:, i, ...])
            total_dice += current_dice
            per_label_dice.append(current_dice.item())
            avg_counter += 1
        # currentDiceLoss = 1 - currentDice # subtract from 1 because this is a loss
    total_dice /= avg_counter

    if per_label:
        return torch.tensor(per_label_dice)
    else:
        return total_dice


def multi_class_dice_per_label(output, label, params):
    """
    This function computes a multi-class dice.

    Args:
        output (torch.Tensor): Input data containing objects. Can be any type but will be converted into binary: background where 0, object everywhere else.
        label (torch.Tensor): Input data containing objects. Can be any type but will be converted into binary: background where 0, object everywhere else.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        list: The list of per-label dice scores.
    """
    return multi_class_dice(output, label, params, per_label=True)


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference. Adapted from https://github.com/loli/medpy/blob/39131b94f0ab5328ab14a874229320efc2f74d98/medpy/metric/binary.py#L1195.

    Args:
        result (torch.Tensor): Input prediction containing objects. Can be any type but will be converted into binary: background where 0, object everywhere else.
        reference (torch.Tensor): Input ground truth containing objects. Can be any type but will be converted into binary: background where 0, object everywhere else.
        voxelspacing (tuple): The size of each voxel, defaults to isotropic spacing of 1mm.
        connectivity (int): The connectivity of regions. See scipy.ndimage.generate_binary_structure for more information.

    Returns:
        float: The symmetric Hausdorff Distance between the object(s) in ```result``` and the object(s) in ```reference```. The distance unit is the same as for the spacing of elements along each dimension, which is usually given in mm.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        return 0
    if 0 == np.count_nonzero(reference):
        return 0

    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(
        reference, structure=footprint, iterations=1
    )

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def _nsd_base(a_to_b, b_to_a, threshold):
    """
    This implementation differs from the official surface dice implementation! These two are not comparable!!!!!
    The normalized surface dice is symmetric, so it should not matter whether a or b is the reference image
    This implementation natively supports 2D and 3D images.

    Args:
        a_to_b (np.ndarray): Surface distances from a to b
        b_to_a (np.ndarray): Surface distances from b to a
        threshold (float): distances below this threshold will be counted as true positives. Threshold is in mm, not voxels!

    Returns:
        float: the normalized surface dice between a and b
    """
    if isinstance(a_to_b, int):
        return 0
    if isinstance(b_to_a, int):
        return 0
    numel_a = len(a_to_b)
    numel_b = len(b_to_a)
    tp_a = np.sum(a_to_b <= threshold) / numel_a
    tp_b = np.sum(b_to_a <= threshold) / numel_b
    fp = np.sum(a_to_b > threshold) / numel_a
    fn = np.sum(b_to_a > threshold) / numel_b
    dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + sys.float_info.min)
    return dc


def _calculator_jaccard(
    inp,
    target,
    params,
    per_label=False,
):
    """
    This function returns sensitivity and specificity.

    Args:
        inp (torch.Tensor): Input prediction containing objects. Can be any type but will be converted into binary: background where 0, object everywhere else.
        target (torch.Tensor): Input ground truth containing objects. Can be any type but will be converted into binary: binary: background where 0, object everywhere else.
        params (dict): The parameter dictionary containing training and data information.
        per_label (bool): Whether to return per-label dice scores.

    Returns:
        float: The Jaccard score between the object(s) in ```inp``` and the object(s) in ```target```.
    """
    result_array = _convert_tensor_to_int_label_array(inp)
    target_array = _convert_tensor_to_int_label_array(target)

    jaccard, avg_counter = 0, 0
    jaccard_per_label = []
    for i in range(0, params["model"]["num_classes"]):
        # this check should only happen during validation
        if i != params["model"]["ignore_label_validation"]:
            dice_score = dice(result_array[:, i, ...], target_array[:, i, ...])
            # https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient#Difference_from_Jaccard
            j_score = dice_score / (2 - dice_score)
            jaccard += j_score
            if per_label:
                jaccard_per_label.append(jaccard)
            avg_counter += 1

    if per_label:
        return torch.tensor(jaccard_per_label)
    else:
        return torch.tensor(jaccard / avg_counter)


def _calculator_sensitivity_specificity(
    inp,
    target,
    params,
    per_label=False,
):
    """
    This function returns sensitivity and specificity.

    Args:
        inp (torch.Tensor): Input prediction containing objects. Can be any type but will be converted into binary: background where 0, object everywhere else.
        target (torch.Tensor): Input ground truth containing objects. Can be any type but will be converted into binary: binary: background where 0, object everywhere else.
        params (dict): The parameter dictionary containing training and data information.
        per_label (bool): Whether to return per-label dice scores.

    Returns:
        float, float: The sensitivity and specificity between the object(s) in ```inp``` and the object(s) in ```target```.
    """
    # inMask is mask of input array equal to a certain tissue (ie. all one's in tumor core)
    # Ref mask is mask of certain tissue in ground truth (ie. all one's in refernce core )
    # overlap is mask where the two equal each other
    # They are of the total number of voxels of the ground truth brain mask

    def get_sensitivity_and_specificity(result_array, target_array):
        iC = np.sum(result_array)
        rC = np.sum(target_array)

        overlap = np.where((result_array == target_array), 1, 0)

        # Where they agree are both equal to that value
        TP = overlap[result_array == 1].sum()
        FP = iC - TP
        FN = rC - TP
        TN = np.count_nonzero((result_array != 1) & (target_array != 1))

        Sens = 1.0 * TP / (TP + FN + sys.float_info.min)
        Spec = 1.0 * TN / (TN + FP + sys.float_info.min)

        # Make Changes if both input and reference are 0 for the tissue type
        if (iC == 0) and (rC == 0):
            Sens = 1.0

        return Sens, Spec

    result_array = _convert_tensor_to_int_label_array(inp)
    target_array = _convert_tensor_to_int_label_array(target)

    sensitivity, specificity, avg_counter = 0, 0, 0
    sensitivity_per_label, specificity_per_label = [], []
    for b in range(0, result_array.shape[0]):
        for i in range(0, params["model"]["num_classes"]):
            if i != params["model"]["ignore_label_validation"]:
                s, p = get_sensitivity_and_specificity(
                    result_array[b, i, ...], target_array[b, i, ...]
                )
                sensitivity += s
                specificity += p
                if per_label:
                    sensitivity_per_label.append(s)
                    specificity_per_label.append(p)
                avg_counter += 1

    if per_label:
        return torch.tensor(sensitivity_per_label), torch.tensor(specificity_per_label)
    else:
        return torch.tensor(sensitivity / avg_counter), torch.tensor(
            specificity / avg_counter
        )


def _calculator_generic_all_surface_distances(
    inp,
    target,
    params,
    per_label=False,
):
    """
    This function returns hd100, hd95, and nsd.

    Args:
        inp (torch.Tensor): Input prediction containing objects. Can be any type but will be converted into binary: background where 0, object everywhere else.
        target (torch.Tensor): Input ground truth containing objects. Can be any type but will be converted into binary: binary: background where 0, object everywhere else.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        float, float, float: The Normalized Surface Dice, 100th percentile Hausdorff Distance, and the 95th percentile Hausdorff Distance.
    """
    result_array = _convert_tensor_to_int_label_array(inp)
    target_array = _convert_tensor_to_int_label_array(target)

    avg_counter = 0
    if per_label:
        return_hd100, return_hd95, return_nsd = [], [], []
    else:
        return_hd100, return_hd95, return_nsd = 0, 0, 0
    for b in range(0, result_array.shape[0]):
        for i in range(0, params["model"]["num_classes"]):
            if i != params["model"]["ignore_label_validation"]:
                hd1 = __surface_distances(
                    result_array[b, i, ...],
                    target_array[b, i, ...],
                    params["subject_spacing"][b],
                )
                hd2 = __surface_distances(
                    target_array[b, i, ...],
                    result_array[b, i, ...],
                    params["subject_spacing"][b],
                )
                threshold = max(min(params["subject_spacing"][0]), 1).item()
                temp_nsd = _nsd_base(hd1, hd2, threshold)
                temp_hd100 = np.percentile(np.hstack((hd1, hd2)), 100)
                temp_hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
                if per_label:
                    return_nsd.append(temp_nsd)
                    return_hd100.append(temp_hd100)
                    return_hd95.append(temp_hd95)
                else:
                    return_nsd += temp_nsd
                    return_hd100 += temp_hd100
                    return_hd95 += temp_hd95
                    avg_counter += 1

    if per_label:
        return (
            torch.tensor(return_nsd),
            torch.tensor(return_hd100),
            torch.tensor(return_hd95),
        )
    else:
        return (
            torch.tensor(return_nsd / avg_counter),
            torch.tensor(return_hd100 / avg_counter),
            torch.tensor(return_hd95 / avg_counter),
        )


def _calculator_generic(
    inp,
    target,
    params,
    percentile=95,
    surface_dice=False,
    per_label=False,
):
    """
    Generic Surface Dice (SD)/Hausdorff (HD) Distance calculation from 2 tensors. Compared to the standard Hausdorff Distance, this metric is slightly more stable to small outliers and is commonly used in Biomedical Segmentation challenges.

    Args:
        inp (torch.Tensor): Input prediction containing objects. Can be any type but will be converted into binary: background where 0, object everywhere else.
        target (torch.Tensor): Input ground truth containing objects. Can be any type but will be converted into binary: binary: background where 0, object everywhere else.
        params (dict): The parameter dictionary containing training and data information.
        percentile (int, optional): The percentile of surface distances to include during HD calculation. Defaults to 95.
        surface_dice (bool, optional): Whether the SD needs to be calculated or not. Defaults to False.
        per_label (bool, optional): Whether the hausdorff needs to be calculated per label or not. Defaults to False.

    Returns:
        float or list: The symmetric Hausdorff Distance or Normalized Surface Distance between the object(s) in ```result``` and the object(s) in ```reference```. The distance unit is the same as for the spacing of elements along each dimension, which is usually given in mm.

    See also:
        :func:`hd`
    """
    _nsd, _hd100, _hd95 = _calculator_generic_all_surface_distances(
        inp, target, params, per_label=per_label
    )
    if surface_dice:
        return _nsd
    elif percentile == 95:
        return _hd95
    else:
        return _hd100


def hd95(inp, target, params):
    return _calculator_generic(inp, target, params, percentile=95)


def hd95_per_label(inp, target, params):
    return _calculator_generic(inp, target, params, percentile=95, per_label=True)


def hd100(inp, target, params):
    return _calculator_generic(inp, target, params, percentile=100)


def hd100_per_label(inp, target, params):
    return _calculator_generic(inp, target, params, percentile=100, per_label=True)


def nsd(inp, target, params):
    return _calculator_generic(inp, target, params, percentile=100, surface_dice=True)


def nsd_per_label(inp, target, params):
    return _calculator_generic(
        inp, target, params, percentile=100, per_label=True, surface_dice=True
    )


def sensitivity(inp, target, params):
    s, _ = _calculator_sensitivity_specificity(inp, target, params)
    return s


def sensitivity_per_label(inp, target, params):
    s, _ = _calculator_sensitivity_specificity(inp, target, params, per_label=True)
    return s


def specificity_segmentation(inp, target, params):
    _, p = _calculator_sensitivity_specificity(inp, target, params)
    return p


def specificity_segmentation_per_label(inp, target, params):
    _, p = _calculator_sensitivity_specificity(inp, target, params, per_label=True)
    return p


def jaccard(inp, target, params):
    j = _calculator_jaccard(inp, target, params)
    return j


def jaccard_per_label(inp, target, params):
    j = _calculator_jaccard(inp, target, params, per_label=True)
    return j
