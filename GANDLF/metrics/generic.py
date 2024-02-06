import torch
from torchmetrics import (
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


def generic_function_output_with_check(predicted_classes, label, metric_function):
    if torch.min(predicted_classes) < 0:
        print(
            "WARNING: Negative values detected in prediction, cannot compute torchmetrics calculations."
        )
        return torch.zeros((1), device=predicted_classes.device)
    else:
        # I need to do this with try-except, otherwise for binary problems it will
        # raise and error as the binary metrics do not have .num_classes
        # attribute.
        # https://github.com/Lightning-AI/torchmetrics/blob/v1.1.2/src/torchmetrics/classification/accuracy.py#L31-L146 link to example from BinaryAccuracy.
        try:
            max_clamp_val = metric_function.num_classes - 1
        except AttributeError:
            max_clamp_val = 1
        predicted_new = torch.clamp(predicted_classes.cpu().int(), max=max_clamp_val)
        predicted_new = predicted_new.reshape(label.shape)
        return metric_function(predicted_new, label.cpu().int())


def generic_torchmetrics_score(output, label, metric_class, metric_key, params):
    task = determine_classification_task_type(params)
    num_classes = params["model"]["num_classes"]
    predicted_classes = output
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    elif params["problem_type"] == "segmentation":
        label = one_hot(label, params["model"]["class_list"])
    metric_function = metric_class(
        task=task,
        num_classes=num_classes,
        threshold=params["metrics"][metric_key]["threshold"],
        average=define_average_type_key(params, metric_key),
        multidim_average=define_multidim_average_type_key(params, metric_key),
    )

    return generic_function_output_with_check(
        predicted_classes.cpu().int(), label.cpu().int(), metric_function
    )


def recall_score(output, label, params):
    return generic_torchmetrics_score(output, label, Recall, "recall", params)


def precision_score(output, label, params):
    return generic_torchmetrics_score(output, label, Precision, "precision", params)


def f1_score(output, label, params):
    return generic_torchmetrics_score(output, label, F1Score, "f1", params)


def accuracy(output, label, params):
    return generic_torchmetrics_score(output, label, Accuracy, "accuracy", params)


def specificity_score(output, label, params):
    return generic_torchmetrics_score(output, label, Specificity, "specificity", params)


def iou_score(output, label, params):
    num_classes = params["model"]["num_classes"]
    predicted_classes = output
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    elif params["problem_type"] == "segmentation":
        label = one_hot(label, params["model"]["class_list"])
    task = determine_classification_task_type(params)
    recall = JaccardIndex(
        task=task,
        num_classes=num_classes,
        average=define_average_type_key(params, "iou"),
        threshold=params["metrics"]["iou"]["threshold"],
    )

    return generic_function_output_with_check(
        predicted_classes.cpu().int(), label.cpu().int(), recall
    )
