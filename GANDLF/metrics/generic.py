import sys
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
from GANDLF.utils.generic import determine_task


def define_average_type_key(params, metric_name):
    """Determine if the metric config defines the type of average to use.
    If not, fallback to the default (macro) average type.
    """
    average_type_key = params["metrics"][metric_name].get("average", "macro")
    return average_type_key


def generic_function_output_with_check(
    predicted_classes, label, metric_function
):
    if torch.min(predicted_classes) < 0:
        print(
            "WARNING: Negative values detected in prediction, cannot compute torchmetrics calculations."
        )
        return torch.zeros((1), device=predicted_classes.device)
    else:
        # this ensures that we always have at least 1 class to clamp to
        # max_clamp_val = min(metric_function.num_classes - 1, 1)
        # I need to do this that way, otherwise for binary problems it will
        # raise and error
        try:
            max_clamp_val = metric_function.num_classes - 1
        except AttributeError:
            max_clamp_val = 1
        predicted_new = torch.clamp(
            predicted_classes.cpu().int(), max=max_clamp_val
        )
        predicted_new = predicted_new.reshape(label.shape)
        return metric_function(predicted_new, label.cpu().int())


def generic_torchmetrics_score(
    output, label, metric_class, metric_key, params
):
    task = determine_task(params)
    num_classes = params["model"]["num_classes"]
    predicted_classes = output
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    elif params["problem_type"] == "segmentation":
        label = one_hot(label, params["model"]["class_list"])
    else:
        params["metrics"][metric_key]["multi_class"] = False
        params["metrics"][metric_key]["mdmc_average"] = None
    metric_function = metric_class(
        task=task,
        average=params["metrics"][metric_key]["average"],
        num_classes=num_classes,
        threshold=params["metrics"][metric_key]["threshold"],
    )

    return generic_function_output_with_check(
        predicted_classes.cpu().int(), label.cpu().int(), metric_function
    )


def recall_score(output, label, params):
    return generic_torchmetrics_score(output, label, Recall, "recall", params)


def precision_score(output, label, params):
    return generic_torchmetrics_score(
        output, label, Precision, "precision", params
    )


def f1_score(output, label, params):
    return generic_torchmetrics_score(output, label, F1Score, "f1", params)


def accuracy(output, label, params):
    return generic_torchmetrics_score(
        output, label, Accuracy, "accuracy", params
    )


def specificity_score(output, label, params):
    return generic_torchmetrics_score(
        output, label, Specificity, "specificity", params
    )


def iou_score(output, label, params):
    num_classes = params["model"]["num_classes"]
    predicted_classes = output
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    elif params["problem_type"] == "segmentation":
        label = one_hot(label, params["model"]["class_list"])
    task = determine_task(params)
    recall = sys.float_info.max
    if task == "binary":
        recall = JaccardIndex(
            task=task,
            threshold=params["metrics"]["iou"]["threshold"],
        )
    elif task == "multiclass":
        recall = JaccardIndex(
            task=task,
            average=define_average_type_key(params, "iou"),
            num_classes=num_classes,
            threshold=params["metrics"]["iou"]["threshold"],
        )
    else:
        print(
            f"IoU score is not implemented for multilabel problems, setting recall to {recall}"
        )

    return generic_function_output_with_check(
        predicted_classes.cpu().int(), label.cpu().int(), recall
    )
