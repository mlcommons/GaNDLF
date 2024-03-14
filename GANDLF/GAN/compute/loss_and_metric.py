import sys
from GANDLF.losses import global_losses_dict
from GANDLF.metrics import global_metrics_dict
from torch import Tensor
from typing import Union, Optional, Tuple


def get_metric_output(
    metric_function: object,
    predicted: Tensor,
    ground_truth: Tensor,
    params: dict,
) -> Union[float, list]:
    """
    This function computes the output of a metric function.

    Args:
        metric_function (object): The metric function to be used.
        predicted (torch.Tensor): The predicted output from the model.
        ground_truth (torch.Tensor): The ground truth label.
        params (dict): The parameters passed by the user yaml.

    Returns:
        Union[float, list]: The computed metric from the label and the prediction.
    """
    metric_output = metric_function(predicted, ground_truth, params).detach().cpu()
    if metric_output.dim() == 0:
        return metric_output.item()
    else:
        temp = metric_output.tolist()
        # this check is needed for precision
        if len(temp) > 1:
            return temp
        else:
            return metric_output.item()


def get_loss_gans(predictions: Tensor, labels: Tensor, params: dict) -> Tensor:
    """
    Compute the loss value for adversatial generative networks.

    Args:
        predictions (torch.Tensor): The predicted output from the model.
        labels (torch.Tensor): The ground truth label.
        params (dict): The parameters passed by the user yaml.

    Returns:
        loss (torch.Tensor): The computed loss from the label and the prediction.
    """

    if isinstance(params["loss_function"], dict):
        # check for mse_torch
        loss_function = global_losses_dict[list(params["loss_function"].keys())[0]]
    else:
        loss_str_lower = params["loss_function"].lower()
        if loss_str_lower in global_losses_dict:
            loss_function = global_losses_dict[loss_str_lower]
        else:
            sys.exit(
                "WARNING: Could not find the requested loss function '"
                + params["loss_function"]
            )

    loss = loss_function(predictions, labels)
    return loss


def get_loss_and_metrics_gans(
    images: Tensor,
    secondary_images: Optional[Tensor],
    labels: Tensor,
    predictions: Tensor,
    params: dict,
) -> Tuple[Tensor, Union[None, dict]]:
    """
    A function to compute the loss and optionally the metrics for generative
    adversarial networks.

    Args:
        images (torch.Tensor): The input image stack according to requirements.
        secondary_images (optional, torch.Tensor): The input secondary image stack
    used only when computing metrics.
        predictions (torch.Tensor) : discriminator output
        labels (torch.Tensor) : ground truth
        params (dict): The parameters passed by the user yaml.

    Returns:
        loss (torch.Tensor): The computed loss from the label and the prediction.
        Union[None, dict]: The computed metric from the label and the prediction.
    Metric is returned only if secondary_images is not None.
    """
    loss = get_loss_gans(predictions, labels, params)
    metric_output = {}
    # Metrics should be a list
    if secondary_images is not None:
        for metric in params["metrics"]:
            metric_lower = metric.lower()
            metric_output[metric] = 0
            if metric_lower in global_metrics_dict:
                metric_function = global_metrics_dict[metric_lower]
                metric_output[metric] = get_metric_output(
                    metric_function, images, secondary_images, params
                )
        return loss, metric_output
    return loss, None
