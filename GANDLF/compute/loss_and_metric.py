import sys
from GANDLF.losses import global_losses_dict
from GANDLF.metrics import global_metrics_dict


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
    if isinstance(
        params["loss_function"], dict
    ):  # this is currently only happening for mse_torch
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

    # specialized loss function for sdnet
    sdnet_check = (len(predicted) > 1) and (params["model"]["architecture"] == "sdnet")
    if sdnet_check:
        # this is specific for sdnet-style archs
        loss_seg = loss_function(predicted[0], ground_truth.squeeze(-1), params)
        loss_reco = global_losses_dict["l1"](predicted[1], image[:, :1, ...], None)
        loss_kld = global_losses_dict["kld"](predicted[2], predicted[3])
        loss_cycle = global_losses_dict["mse"](predicted[2], predicted[4], None)
        loss = 0.01 * loss_kld + loss_reco + 10 * loss_seg + loss_cycle
    else:
        loss = loss_function(predicted, ground_truth, params)
    metric_output = {}

    # Metrics should be a list
    for metric in params["metrics"]:
        metric_lower = metric.lower()
        if metric_lower in global_metrics_dict:
            metric_function = global_metrics_dict[metric_lower]
            if sdnet_check:
                metric_output[metric] = (
                    metric_function(predicted[0], ground_truth.squeeze(-1), params)
                    .cpu()
                    .data.item()
                )
            else:
                metric_output[metric] = (
                    metric_function(predicted, ground_truth, params).cpu().data.item()
                )
        else:
            print(
                "WARNING: Could not find the requested metric '" + metric,
                file=sys.stderr,
            )
    return loss, metric_output
