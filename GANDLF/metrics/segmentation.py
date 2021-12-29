"""
All the segmentation metrics are to be called from here
"""
import torch, numpy
from GANDLF.losses.segmentation import dice
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import (
    distance_transform_edt,
    binary_erosion,
    generate_binary_structure,
)


def multi_class_dice(output, label, params, per_label=False):
    """
    This function computes a multi-class dice

    Parameters
    ----------
    output : torch.Tensor
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    label : torch.Tensor
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    params : dict
        The parameter dictionary containing training and data information.
    per_label : bool
        Whether the dice needs to be calculated per label or not.

    Returns
    -------
    total_dice : float or list
        The average dice for all labels or a list of per-label dice scores.

    """
    total_dice = 0
    avg_counter = 0
    per_label_dice = []
    for i in range(0, params["model"]["num_classes"]):
        # this check should only happen during validation
        if i != params["model"]["ignore_label_validation"]:
            current_dice = dice(output[:, i, ...], label[:, i, ...])
            total_dice += current_dice
            per_label_dice.append(current_dice)
            avg_counter += 1
        # currentDiceLoss = 1 - currentDice # subtract from 1 because this is a loss
    total_dice /= avg_counter

    if per_label:
        return torch.tensor(per_label_dice)
    else:
        return total_dice


def multi_class_dice_per_label(output, label, params):
    """
    This function computes a multi-class dice

    Parameters
    ----------
    output : torch.Tensor
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    label : torch.Tensor
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    params : dict
        The parameter dictionary containing training and data information.

    Returns
    -------
    total_dice : list
        The list of per-label dice scores.
    """
    return multi_class_dice(output, label, params, per_label=True)


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.

    Adopted from https://github.com/loli/medpy/blob/39131b94f0ab5328ab14a874229320efc2f74d98/medpy/metric/binary.py#L1195
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = numpy.asarray(voxelspacing, dtype=numpy.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == numpy.count_nonzero(result):
        # print(
        #     "The first supplied array does not contain any binary object.",
        #     file=sys.stderr,
        # )
        return 0
    if 0 == numpy.count_nonzero(reference):
        # print(
        #     "The second supplied array does not contain any binary object.",
        #     file=sys.stderr,
        # )
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


def hd_generic(inp, target, params, percentile=95, per_label=False):
    """
    Generic Hausdorff Distance calculation
    Computes the Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.
    Parameters
    ----------
    result : torch.tensor
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : torch.tensor
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    params : dict
        The parameter dictionary containing training and data information.
    percentile : int
        The percentile of surface distances to include during Hausdorff calculation.
    per_label : bool
        Whether the hausdorff needs to be calculated per label or not.
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.
    See also
    --------
    :func:`hd`
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result_array = inp.detach().cpu().numpy()
    target_array = target.detach().cpu().numpy()
    if result_array.shape[-1] == 1:
        result_array = result_array.squeeze(-1)
    if target_array.shape[-1] == 1:
        target_array = target_array.squeeze(-1)
    # ensure that we are dealing with a binary array
    result_array[result_array < 0.5] = 0
    result_array[result_array >= 0.5] = 1

    hd = 0
    avg_counter = 0
    hd_per_label = []
    for b in range(0, result_array.shape[0]):
        for i in range(0, params["model"]["num_classes"]):
            if i != params["model"]["ignore_label_validation"]:
                hd1 = __surface_distances(
                    result_array[b, i, ...],
                    target_array[b, i, ...],
                )
                hd2 = __surface_distances(
                    target_array[b, i, ...],
                    result_array[b, i, ...],
                    params["subject_spacing"][b],
                )
                current_hd = numpy.percentile(numpy.hstack((hd1, hd2)), percentile)
                hd += current_hd
                hd_per_label.append(current_hd)
                avg_counter += 1

    if per_label:
        return torch.tensor(hd_per_label)
    else:
        return torch.tensor(hd / avg_counter)


def hd95(inp, target, params):
    return hd_generic(inp, target, params, 95)


def hd95_per_label(inp, target, params):
    return hd_generic(inp, target, params, 95, True)


def hd100(inp, target, params):
    return hd_generic(inp, target, params, 100)


def hd100_per_label(inp, target, params):
    return hd_generic(inp, target, params, 100, True)
