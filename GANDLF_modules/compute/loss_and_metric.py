import sys
from GANDLF.losses import global_losses_dict
from GANDLF.metrics import global_metrics_dict
import torch.nn.functional as nnf

from GANDLF.utils import one_hot, reverse_one_hot, get_linear_interpolation_mode


def get_metric_output(metric_function, predicted, ground_truth, params):
    """
    This function computes the output of a metric function.
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


def get_loss_and_metrics(image, ground_truth, predicted, params):
    """
    This function computes the loss and metrics for a given image, ground truth and predicted output.

    Args:
        image (torch.Tensor): The input image stack according to requirements.
        ground_truth (torch.Tensor): The input ground truth for the corresponding image label.
        predicted (torch.Tensor): The input predicted label for the corresponding image label.
        params (dict): The parameters passed by the user yaml.

    Returns:
        torch.Tensor: The computed loss from the label and the prediction.
        dict: The computed metric from the label and the prediction.
    """
    # this is currently only happening for mse_torch
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

    loss = 0
    # specialized loss function for sdnet
    sdnet_check = (len(predicted) > 1) and (params["model"]["architecture"] == "sdnet")

    if params["problem_type"] == "segmentation":
        ground_truth = one_hot(ground_truth, params["model"]["class_list"])

    deep_supervision_model = False
    if (
        (len(predicted) > 1)
        and not (sdnet_check)
        and ("deep" in params["model"]["architecture"])
    ):
        deep_supervision_model = True
        # this case is for models that have deep-supervision - currently only used for segmentation models
        # these weights are taken from previous publication (https://arxiv.org/pdf/2103.03759.pdf)
        loss_weights = [0.5, 0.25, 0.175, 0.075]

        assert len(predicted) == len(
            loss_weights
        ), "Loss weights must be same length as number of outputs."

        ground_truth_resampled = []
        ground_truth_prev = ground_truth.detach()
        for i, _ in enumerate(predicted):
            if ground_truth_prev[0].shape != predicted[i][0].shape:
                # we get the expected shape of resampled ground truth
                expected_shape = reverse_one_hot(
                    predicted[i][0].detach(), params["model"]["class_list"]
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
        loss_seg = loss_function(predicted[0], ground_truth.squeeze(-1), params)
        loss_reco = global_losses_dict["l1"](predicted[1], image[:, :1, ...], None)
        loss_kld = global_losses_dict["kld"](predicted[2], predicted[3])
        loss_cycle = global_losses_dict["mse"](predicted[2], predicted[4], None)
        loss = 0.01 * loss_kld + loss_reco + 10 * loss_seg + loss_cycle
    else:
        if deep_supervision_model:
            # this is for models that have deep-supervision
            for i, _ in enumerate(predicted):
                # loss is calculated based on resampled "soft" labels using a pre-defined weights array
                loss += (
                    loss_function(predicted[i], ground_truth_resampled[i], params)
                    * loss_weights[i]
                )
        else:
            loss = loss_function(predicted, ground_truth, params)
    metric_output = {}

    # Metrics should be a list
    for metric in params["metrics"]:
        metric_lower = metric.lower()
        metric_output[metric] = 0
        if metric_lower in global_metrics_dict:
            metric_function = global_metrics_dict[metric_lower]
            if sdnet_check:
                metric_output[metric] = get_metric_output(
                    metric_function, predicted[0], ground_truth.squeeze(-1), params
                )
            else:
                if deep_supervision_model:
                    for i, _ in enumerate(predicted):
                        metric_output[metric] += get_metric_output(
                            metric_function,
                            predicted[i],
                            ground_truth_resampled[i],
                            params,
                        )

                else:
                    metric_output[metric] = get_metric_output(
                        metric_function, predicted, ground_truth, params
                    )
    return loss, metric_output
