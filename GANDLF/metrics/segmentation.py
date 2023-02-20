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
        float or list: The symmetric Hausdorff Distance between the object(s) in ```result``` and the object(s) in ```reference```. The distance unit is the same as for the spacing of elements along each dimension, which is usually given in mm.

    See also:
        :func:`hd`
    """
    result_array = _convert_tensor_to_int_label_array(inp)
    target_array = _convert_tensor_to_int_label_array(target)

    hd = 0
    avg_counter = 0
    hd_per_label = []
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
                if surface_dice:
                    # ensure threshold always at least 1
                    threshold = max(min(params["subject_spacing"][0]), 1).item()
                    current_hd = _nsd_base(hd1, hd2, threshold)
                else:
                    current_hd = np.percentile(np.hstack((hd1, hd2)), percentile)
                hd += current_hd
                hd_per_label.append(current_hd)
                avg_counter += 1

    if per_label:
        return torch.tensor(hd_per_label)
    else:
        return torch.tensor(hd / avg_counter)


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
