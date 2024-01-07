import torch
import torchmetrics as tm
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union


def overall_stats(
    generated_images: torch.Tensor,
    real_images: torch.Tensor,
    params: Dict[str, Any],
):
    """
    Generates a dictionary of metrics calculated on the overall generated
    images and real images.
    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        params (dict): The parameter dictionary containing training and data
    information.
    Returns:
        dict: A dictionary of metrics.
    """
    assert (
        params["problem_type"] == "synthesis"
    ), "Only synthesis is supported for these stats"
    output_metrics = {}
    reduction_types_keys = {
        "elementwise_mean": "elementwise_mean",
        "none": "none",
        "sum": "sum",
    }
    # TODO


def _calculator_ssim(
    generated_images: torch.Tensor,
    real_images: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """
    This function computes the SSIM between the generated images and the real
    images. Except for the params specified below, the rest of the params are
    default from torchmetrics.
    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        reduction (str): The reduction type.
    Returns:
        torch.Tensor: The SSIM score.
    """
    ssim = tm.image.StructuralSimilarityIndexMeasure(reduction=reduction)
    return ssim(generated_images, real_images)


def _calculator_FID(
    generated_images: torch.Tensor,
    real_images: torch.Tensor,
    n_input_channels: int,
) -> torch.Tensor:
    """This function computes the FID between the generated images and the
    real images. Except for the params specified below, the rest of the params
    are default from torchmetrics.
    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        n_input_channels (int): The number of input channels.
    Returns:
        torch.Tensor: The FID score.
    """
    fid_metric = tm.image.fid.FrechetInceptionDistance(
        feature=1024,
        normalize=True,
    )
    if n_input_channels == 1:
        # need manual patching for single channel data
        fid_metric.get_submodule("inception")._modules[
            "Conv2d_1a_3x3"
        ]._modules["conv"] = torch.nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False
        )
    # check input dtype
    if generated_images.dtype != torch.float32:
        generated_images = generated_images.float()
    if real_images.dtype != torch.float32:
        real_images = real_images.float()
    if generated_images.max() > 1:
        warnings.warn(
            "Input generated images are not in [0, 1] range. "
            "This may lead to incorrect results. "
            "FID expects input images to be in [0, 1] range."
            "Dividing the images by 255 for metric calculation."
        )
        generated_images = generated_images / 255.0
    if real_images.max() > 1:
        warnings.warn(
            "Input real images are not in [0, 1] range. "
            "This may lead to incorrect results. "
            "FID expects input images to be in [0, 1] range."
            "Dividing the images by 255 for metric calculation."
        )
        real_images = real_images / 255.0
    fid_metric.update(generated_images, real=False)
    fid_metric.update(real_images, real=True)
    return fid_metric.compute()


def _calculator_LPIPS(
    generated_images: torch.Tensor,
    real_images: torch.Tensor,
    reduction: str,
    n_input_channels: int,
) -> torch.Tensor:
    """This function computes the LPIPS between the generated images and the
    real images. Except for the params specified below, the rest of the params
    are default from torchmetrics.
    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        reduction (str): The reduction type.
        n_input_channels (int): The number of input channels.
    Returns:
        torch.Tensor: The LPIP score.
    """
    lpips_metric = tm.image.lpip.LearnedPerceptualImagePatchSimilarity(
        net="squeeze",
        normalize=True,
        reduction=reduction,
    )
    if n_input_channels == 1:
        # need manual patching for single channel data
        lpips_metric.net.net.slices._modules["0"]._modules[
            "0"
        ] = torch.nn.Conv2d(
            1,
            64,
            kernel_size=(3, 3),
            stride=(2, 2),
        )
    # check input dtype
    if generated_images.dtype != torch.float32:
        generated_images = generated_images.float()
    if real_images.dtype != torch.float32:
        real_images = real_images.float()
    if generated_images.max() > 1:
        warnings.warn(
            "Input generated images are not in [0, 1] range. "
            "This may lead to incorrect results. "
            "LPIPS expects input images to be in [0, 1] range."
            "Dividing the images by 255 for metric calculation."
        )
        generated_images = generated_images / 255.0
    if real_images.max() > 1:
        warnings.warn(
            "Input real images are not in [0, 1] range. "
            "This may lead to incorrect results. "
            "LPIPS expects input images to be in [0, 1] range."
            "Dividing the images by 255 for metric calculation."
        )
        real_images = real_images / 255.0
    return lpips_metric(generated_images, real_images)
