import torch
from torchmetrics import (
    Metric,
    F1Score,
    Precision,
    Recall,
    JaccardIndex,
    Accuracy,
    Specificity,
)
from GANDLF.utils.tensor import one_hot
from GANDLF.utils.generic import (
    determine_classification_task_type,
    define_average_type_key,
    define_multidim_average_type_key,
)


def generic_function_output_with_check(
    prediction: torch.Tensor, target: torch.Tensor, metric_function: object
) -> torch.Tensor:
    """
    This function computes the output of a generic metric function.

    Args:
        prediction (torch.Tensor): The prediction of the model.
        target (torch.Tensor): The ground truth labels.
        metric_function (object): The metric function to be used, which is a wrapper around the torchmetrics class.

    Returns:
        torch.Tensor: The output of the metric function.
    """
    if torch.min(prediction) < 0:
        print(
            "WARNING: Negative values detected in prediction, cannot compute torchmetrics calculations."
        )
        return torch.tensor(0, device=prediction.device)
    else:
        # I need to do this with try-except, otherwise for binary problems it will
        # raise and error as the binary metrics do not have .num_classes
        # attribute.
        # https://github.com/Lightning-AI/torchmetrics/blob/v1.1.2/src/torchmetrics/classification/accuracy.py#L31-L146 link to example from BinaryAccuracy.
        try:
            max_clamp_val = metric_function.num_classes - 1
        except AttributeError:
            max_clamp_val = 1
        predicted_new = torch.clamp(prediction.cpu().int(), max=max_clamp_val)
        predicted_new = predicted_new.reshape(target.shape)
        return metric_function(predicted_new, target.cpu().int())


def generic_torchmetrics_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    metric_class: Metric,
    metric_key: str,
    params: dict,
) -> torch.Tensor:
    """
    This function computes the output of a generic torchmetrics metric.

    Args:
        prediction (torch.Tensor): The prediction of the model.
        target (torch.Tensor): The ground truth labels.
        metric_class (Metric): The metric class to be used.
        metric_key (str): The key for the metric.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        torch.Tensor: The output of the metric function.
    """
    task = determine_classification_task_type(params)
    num_classes = params["model"]["num_classes"]
    predicted_classes = prediction
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(prediction, 1)
    elif params["problem_type"] == "segmentation":
        target = one_hot(target, params["model"]["class_list"])
    metric_function = metric_class(
        task=task,
        num_classes=num_classes,
        threshold=params["metrics"][metric_key]["threshold"],
        average=define_average_type_key(params, metric_key),
        multidim_average=define_multidim_average_type_key(params, metric_key),
    )

    return generic_function_output_with_check(
        predicted_classes.cpu().int(), target.cpu().int(), metric_function
    )


def recall_score(
    prediction: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    return generic_torchmetrics_score(prediction, target, Recall, "recall", params)


def precision_score(
    prediction: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    return generic_torchmetrics_score(
        prediction, target, Precision, "precision", params
    )


def f1_score(
    prediction: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    return generic_torchmetrics_score(prediction, target, F1Score, "f1", params)


def accuracy(
    prediction: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    return generic_torchmetrics_score(prediction, target, Accuracy, "accuracy", params)


def specificity_score(
    prediction: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    return generic_torchmetrics_score(
        prediction, target, Specificity, "specificity", params
    )


def iou_score(
    prediction: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    num_classes = params["model"]["num_classes"]
    predicted_classes = prediction
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(prediction, 1)
    elif params["problem_type"] == "segmentation":
        target = one_hot(target, params["model"]["class_list"])
    task = determine_classification_task_type(params)
    recall = JaccardIndex(
        task=task,
        num_classes=num_classes,
        average=define_average_type_key(params, "iou"),
        threshold=params["metrics"]["iou"]["threshold"],
    )

    return generic_function_output_with_check(
        predicted_classes.cpu().int(), target.cpu().int(), recall
    )
