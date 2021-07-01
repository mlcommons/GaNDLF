"""
All the metrics are to be called from here
"""
import sys, torch, numpy
from .losses import MSE, MSE_loss
from .utils import one_hot
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import (
    distance_transform_edt,
    binary_erosion,
    generate_binary_structure,
)


# Dice scores and dice losses
def dice(output, label):
    """
    This function computes a dice score between two tensors

    Parameters
    ----------
    output : Tensor
        Output predicted generally by the network
    label : Tensor
        Required target label to match the output with

    Returns
    -------
    Tensor
        Computed Dice Score

    """
    smooth = 1e-7
    iflat = output.contiguous().view(-1)
    tflat = label.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def multi_class_dice(output, label, params):
    """
    This function computes a multi-class dice

    Parameters
    ----------
    output : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.
    num_class : TYPE
        DESCRIPTION.
    weights : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    total_dice : TYPE
        DESCRIPTION.

    """
    label = one_hot(label, params["model"]["class_list"])
    total_dice = 0
    num_class = params["model"]["num_classes"]
    # print("Number of classes : ", params["model"]["num_classes"])
    for i in range(0, num_class):  # 0 is background
        # this check should only happen during validation
        if num_class != params["model"]["ignore_label_validation"]:
            total_dice += dice(output[:, i, ...], label[:, i, ...])
        # currentDiceLoss = 1 - currentDice # subtract from 1 because this is a loss
    total_dice /= num_class
    return total_dice


def accuracy(output, label, params):
    """
    Calculates the accuracy between output and a label

    Parameters
    ----------
    output : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.
    thresh : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if params["metrics"]["accuracy"]["threshold"] is not None:
        output = (output >= params["metrics"]["accuracy"]["threshold"]).float()
    correct = (output == label).float().sum()
    return correct / len(label)


def MSE_loss_agg(inp, target, params):
    return MSE_loss(inp, target, params)


def identity(output, label, params):
    """
    Always returns 0

    Parameters
    ----------
    output : Tensor
        Output predicted generally by the network
    label : Tensor
        Required target label to match the output with

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    _, _, _ = output, label, params
    return torch.Tensor(0)


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
        print(
            "The first supplied array does not contain any binary object.",
            file=sys.stderr,
        )
        return 0
    if 0 == numpy.count_nonzero(reference):
        print(
            "The second supplied array does not contain any binary object.",
            file=sys.stderr,
        )
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


def hd_generic(inp, target, params, percentile=95):
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
    result_array = inp.cpu().detach().numpy()
    if result_array.shape[-1] == 1:
        result_array = result_array.squeeze(-1)
    # ensure that we are dealing with a binary array
    result_array[result_array < 0.5] = 0
    result_array[result_array >= 0.5] = 1
    reference_array = (
        one_hot(target, params["model"]["class_list"]).squeeze(-1).cpu().numpy()
    )

    hd1 = __surface_distances(result_array, reference_array, params["subject_spacing"])
    hd2 = __surface_distances(reference_array, result_array, params["subject_spacing"])
    hd = numpy.percentile(numpy.hstack((hd1, hd2)), percentile)
    return torch.tensor(hd)


def hd95(inp, target, params):
    return hd_generic(inp, target, params, 95)


def hd100(inp, target, params):
    return hd_generic(inp, target, params, 100)


def fetch_metric(metric_name):
    """

    Parameters
    ----------
    metric_name : string
        Should be a name of a metric

    Returns
    -------
    metric_function : function
        The function to compute the metric

    """
    # if dict, only pick the first value
    if isinstance(metric_name, dict):
        metric_name = list(metric_name)[0]

    metric_lower = metric_name.lower()

    if metric_lower == "dice":
        metric_function = multi_class_dice
    elif metric_lower == "accuracy":
        metric_function = accuracy
    elif metric_lower == "mse":
        metric_function = MSE_loss_agg
    elif (metric_lower == "hd95") or (metric_lower == "hausdorff95"):
        metric_function = hd95
    elif (metric_lower == "hd") or (metric_lower == "hausdorff"):
        metric_function = hd100
    else:
        print("Metric was undefined")
        metric_function = identity
    return metric_function
