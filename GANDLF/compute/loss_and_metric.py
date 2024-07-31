import sys
import warnings
from typing import Dict, Tuple, Union
from GANDLF.losses import global_losses_dict
from GANDLF.metrics import global_metrics_dict
import torch
import torch.nn.functional as nnf

from GANDLF.utils import one_hot, reverse_one_hot, get_linear_interpolation_mode


def get_metric_output(
    metric_function: object,
    prediction: torch.Tensor,
    target: torch.Tensor,
    params: dict,
) -> Union[float, list]:
    """
    This function computes the metric output for a given metric function, prediction and target.

    Args:
        metric_function (object): The metric function to be used.
        prediction (torch.Tensor): The input prediction label for the corresponding image label.
        target (torch.Tensor): The input ground truth for the corresponding image label.
        params (dict): The parameters passed by the user yaml.

    Returns:
        float: The computed metric from the label and the prediction.
    """
    metric_output = metric_function(prediction, target, params).detach().cpu()

    if metric_output.dim() == 0:
        return metric_output.item()
    else:
        temp = metric_output.tolist()
        # this check is needed for precision
        if len(temp) > 1:
            return temp
        else:
            # TODO: this branch is extremely age case and is buggy.
            #  Overall the case when metric returns a list but of length 1 is very rare. The only case is when
            #  the metric returns Nx.. tensor (i.e. without aggregation by elements) and batch_size==N==1. This branch
            #  would definitely fail for such a metrics like
            #  MulticlassAccuracy(num_classes=3, multidim_average="samplewise")
            #  Maybe the best solution is to raise an error here if metric is configured to return samplewise results?
            return metric_output.item()


def get_loss_and_metrics(
    image: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor, params: dict
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    This function computes the loss and metrics for a given image, ground truth and prediction output.

    Args:
        image (torch.Tensor): The input image stack according to requirements.
        target (torch.Tensor): The input ground truth for the corresponding image label.
        prediction (torch.Tensor): The input prediction label for the corresponding image label.
        params (dict): The parameters passed by the user yaml.

    Returns:
        Tuple[torch.Tensor, Dict[str,float]]: The computed loss and metrics from the label and the prediction.
    """
    # this is currently only happening for mse_torch
    if isinstance(params["loss_function"], dict):
        # check for mse_torch
        loss_function = global_losses_dict[list(params["loss_function"].keys())[0]]
    else:
        loss_str_lower = params["loss_function"].lower()
        assert (
            loss_str_lower in global_losses_dict
        ), f"Could not find the requested loss function '{params['loss_function']}'"
        loss_function = global_losses_dict[loss_str_lower]

    loss = 0
    # specialized loss function for sdnet
    sdnet_check = (len(prediction) > 1) and (params["model"]["architecture"] == "sdnet")

    if params["problem_type"] == "segmentation":
        target = one_hot(target, params["model"]["class_list"])

    deep_supervision_model = False
    if (
        (len(prediction) > 1)
        and not (sdnet_check)
        and ("deep" in params["model"]["architecture"])
    ):
        deep_supervision_model = True
        # this case is for models that have deep-supervision - currently only used for segmentation models
        # these weights are taken from previous publication (https://arxiv.org/pdf/2103.03759.pdf)
        loss_weights = [0.5, 0.25, 0.175, 0.075]

        assert len(prediction) == len(
            loss_weights
        ), "Loss weights must be same length as number of outputs."

        ground_truth_resampled = []
        ground_truth_prev = target.detach()
        for i, _ in enumerate(prediction):
            if ground_truth_prev[0].shape != prediction[i][0].shape:
                # we get the expected shape of resampled ground truth
                expected_shape = reverse_one_hot(
                    prediction[i][0].detach(), params["model"]["class_list"]
                ).shape

                # linear interpolation is needed because we want "soft" images for resampled ground truth
                ground_truth_prev = nnf.interpolate(
                    ground_truth_prev,
                    size=expected_shape,
                    mode=get_linear_interpolation_mode(len(expected_shape)),
                    align_corners=False,
                )
            ground_truth_resampled.append(ground_truth_prev)

    if sdnet_check:
        # this is specific for sdnet-style archs
        loss_seg = loss_function(prediction[0], target.squeeze(-1), params)
        loss_reco = global_losses_dict["l1"](prediction[1], image[:, :1, ...], None)
        loss_kld = global_losses_dict["kld"](prediction[2], prediction[3])
        loss_cycle = global_losses_dict["mse"](prediction[2], prediction[4], None)
        loss = 0.01 * loss_kld + loss_reco + 10 * loss_seg + loss_cycle
    elif deep_supervision_model:
        # this is for models that have deep-supervision
        for i, _ in enumerate(prediction):
            # loss is calculated based on resampled "soft" labels using a pre-defined weights array
            loss += (
                loss_function(prediction[i], ground_truth_resampled[i], params)
                * loss_weights[i]
            )
    else:
        loss = loss_function(prediction, target, params)
    metric_output = {}

    # Metrics should be a list
    for metric in params["metrics"]:
        metric_lower = metric.lower()
        metric_output[metric] = 0
        if metric_lower not in global_metrics_dict:
            warnings.warn("WARNING: Could not find the requested metric '" + metric)
            continue

        metric_function = global_metrics_dict[metric_lower]
        if sdnet_check:
            metric_output[metric] = get_metric_output(
                metric_function, prediction[0], target.squeeze(-1), params
            )
        elif deep_supervision_model:
            for i, _ in enumerate(prediction):
                metric_output[metric] += get_metric_output(
                    metric_function, prediction[i], ground_truth_resampled[i], params
                )
        else:
            metric_output[metric] = get_metric_output(
                metric_function, prediction, target, params
            )
    return loss, metric_output
