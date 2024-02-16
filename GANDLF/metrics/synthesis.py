from typing import Optional
import SimpleITK as sitk
import PIL.Image
import numpy as np
import torch
from torchmetrics import (
    StructuralSimilarityIndexMeasure,
    MeanSquaredError,
    MeanSquaredLogError,
    MeanAbsoluteError,
    PeakSignalNoiseRatio,
)
from GANDLF.utils import get_image_from_tensor


def structural_similarity_index(
    prediction: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes the structural similarity index between the target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.
        mask (Optional[torch.Tensor], optional): The mask tensor. Defaults to None.

    Returns:
        torch.Tensor: The structural similarity index.
    """
    ssim = StructuralSimilarityIndexMeasure(return_full_image=True)
    _, ssim_idx_full_image = ssim(preds=prediction, target=target)
    mask = torch.ones_like(ssim_idx_full_image) if mask is None else mask
    try:
        ssim_idx = ssim_idx_full_image[mask]
    except Exception as e:
        print(f"Error: {e}")
        if len(ssim_idx_full_image.shape) == 0:
            ssim_idx = torch.ones_like(mask) * ssim_idx_full_image
    return ssim_idx.mean()


def mean_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean squared error between the target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.
    """
    mse = MeanSquaredError()
    return mse(preds=prediction, target=target)


def peak_signal_noise_ratio(
    target: torch.Tensor,
    prediction: torch.Tensor,
    data_range: Optional[tuple] = None,
    epsilon: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes the peak signal to noise ratio between the target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.
        data_range (Optional[tuple], optional): The data range. Defaults to None.
        epsilon (Optional[float], optional): The epsilon value. Defaults to None.

    Returns:
        torch.Tensor: The peak signal to noise ratio.
    """
    if epsilon == None:
        psnr = (
            PeakSignalNoiseRatio()
            if data_range == None
            else PeakSignalNoiseRatio(data_range=data_range[1] - data_range[0])
        )
        return psnr(preds=prediction, target=target)
    else:  # implementation of PSNR that does not give 'inf'/'nan' when 'mse==0'
        mse = mean_squared_error(target, prediction)
        if data_range == None:  # compute data_range like torchmetrics if not given
            min_v = (
                0 if torch.min(target) > 0 else torch.min(target)
            )  # look at this line
            max_v = torch.max(target)
        else:
            min_v, max_v = data_range
        return 10.0 * torch.log10(((max_v - min_v) ** 2) / (mse + epsilon))


def mean_squared_log_error(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Computes the mean squared log error between the target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        torch.Tensor: The mean squared log error.
    """
    mle = MeanSquaredLogError()
    return mle(preds=prediction, target=target)


def mean_absolute_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean absolute error between the target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        torch.Tensor: The mean absolute error.
    """
    mae = MeanAbsoluteError()
    return mae(preds=prediction, target=target)


def _get_ncc_image(prediction: torch.Tensor, target: torch.Tensor) -> sitk.Image:
    """
    Computes normalized cross correlation image between target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        sitk.Image: The normalized cross correlation image.
    """

    def __convert_to_grayscale(image: sitk.Image) -> sitk.Image:
        """
        Helper function to convert image to grayscale.

        Args:
            image (sitk.Image): The image to convert.

        Returns:
            sitk.Image: The converted image.
        """
        if "vector" in image.GetPixelIDTypeAsString().lower():
            temp_array = sitk.GetArrayFromImage(image)
            image_pil = PIL.Image.fromarray(
                np.moveaxis(temp_array[0, ...], 0, 2).astype(np.uint8)
            )
            image_pil_gray = image_pil.convert("L")
            return sitk.GetImageFromArray(image_pil_gray)
        else:
            return image

    target_image = __convert_to_grayscale(get_image_from_tensor(target))
    pred_image = __convert_to_grayscale(get_image_from_tensor(prediction))
    correlation_filter = sitk.FFTNormalizedCorrelationImageFilter()
    return correlation_filter.Execute(target_image, pred_image)


def ncc_mean(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes normalized cross correlation mean between target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The normalized cross correlation mean.
    """
    stats_filter = sitk.StatisticsImageFilter()
    corr_image = _get_ncc_image(target, prediction)
    stats_filter.Execute(corr_image)
    return stats_filter.GetMean()


def ncc_std(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes normalized cross correlation standard deviation between target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The normalized cross correlation standard deviation.
    """
    stats_filter = sitk.StatisticsImageFilter()
    corr_image = _get_ncc_image(target, prediction)
    stats_filter.Execute(corr_image)
    return stats_filter.GetSigma()


def ncc_max(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes normalized cross correlation maximum between target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The normalized cross correlation maximum.
    """
    stats_filter = sitk.StatisticsImageFilter()
    corr_image = _get_ncc_image(target, prediction)
    stats_filter.Execute(corr_image)
    return stats_filter.GetMaximum()


def ncc_min(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes normalized cross correlation minimum between target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The normalized cross correlation minimum.
    """
    stats_filter = sitk.StatisticsImageFilter()
    corr_image = _get_ncc_image(target, prediction)
    stats_filter.Execute(corr_image)
    return stats_filter.GetMinimum()
