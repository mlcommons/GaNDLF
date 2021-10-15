import sys
from GANDLF.losses import global_losses_dict
from GANDLF.metrics import global_metrics_dict
import torch.nn.functional as nnf

from GANDLF.utils.tensor import one_hot, reverse_one_hot


def get_loss_and_metrics(image, ground_truth, predicted, params):
    """
    image: torch.Tensor
        The input image stack according to requirements
    ground_truth : torch.Tensor
        The input ground truth for the corresponding image label
    predicted : torch.Tensor
        The input predicted label for the corresponding image label
    params : dict
        The parameters passed by the user yaml

    Returns
    -------
    loss : torch.Tensor
        The computed loss from the label and the output
    metric_output : torch.Tensor
        The computed metric from the label and the output
    """
    # this is currently only happening for mse_torch
    if isinstance(params["loss_function"], dict):
        # check for mse_torch
        loss_function = global_losses_dict["mse"]
    else:
        loss_str_lower = params["loss_function"].lower()
        if loss_str_lower in global_losses_dict:
            loss_function = global_losses_dict[loss_str_lower]
        else:
            sys.exit(
                "WARNING: Could not find the requested loss function '"
                + params["loss_function"],
                file=sys.stderr,
            )

    loss = 0
    # specialized loss function for sdnet
    sdnet_check = (len(predicted) > 1) and (params["model"]["architecture"] == "sdnet")

    if params["problem_type"] == "segmentation":
        ground_truth = one_hot(ground_truth, params["model"]["class_list"])

    if (
        (len(predicted) > 1)
        and not (sdnet_check)
        and (params["problem_type"] == "segmentation")
    ):
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

                # apparently, we need to pass tri- or bi- linear for nnf.interpolate because "linear" doesn't work
                # linear interpolation is needed because we want "soft" images for resampled ground truth
                mode = "nearest"
                if len(expected_shape) == 3:
                    mode = "trilinear"
                elif len(expected_shape) == 2:
                    mode = "bilinear"
                ground_truth_prev = nnf.interpolate(
                    ground_truth_prev, size=expected_shape, mode=mode
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
        if len(predicted) > 1:
            for i, _ in enumerate(predicted):
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
                metric_output[metric] = (
                    metric_function(predicted[0], ground_truth.squeeze(-1), params)
                    .detach()
                    .cpu()
                    .data.item()
                )
            else:
                if len(predicted) > 1:
                    for i, _ in enumerate(predicted):
                        metric_output[metric] += (
                            metric_function(
                                predicted[i], ground_truth_resampled[i], params
                            )
                            .detach()
                            .cpu()
                            .item()
                        )

                else:
                    metric_output[metric] = (
                        metric_function(predicted, ground_truth, params)
                        .detach()
                        .cpu()
                        .item()
                    )
        else:
            print(
                "WARNING: Could not find the requested metric '" + metric,
                file=sys.stderr,
            )
    return loss, metric_output
