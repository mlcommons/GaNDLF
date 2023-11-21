import sys
import torch
import warnings
from typing import Dict, Union, Any
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


def define_average_type_key(
    params: Dict[str, Union[Dict[str, Any], Any]], metric_name: str
):
    """Determine if the the 'average' filed is defined in the metric config.
    If not, fallback to the default 'macro'
    values.
    Args:
        params (dict): The parameter dictionary containing training and data information.
        metric_name (str): The name of the metric.
    Returns:
        str: The average type key.
    """
    average_type_key = params["metrics"][metric_name].get("average", "macro")
    return average_type_key


def define_multidim_average_type_key(params, metric_name):
    """Determine if the the 'multidim_average' filed is defined in the metric config.
    If not, fallback to the default 'global'.
    Args:
        params (dict): The parameter dictionary containing training and data information.
        metric_name (str): The name of the metric.
    Returns:
        str: The average type key.
    """
    average_type_key = params["metrics"][metric_name].get(
        "multidim_average", "global"
    )
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
        # I need to do this that way, otherwise for binary problems it will
        # raise and error as the binary metrics do not have .num_classes
        # attribute.
        # https://tinyurl.com/564rh9yp link to example from BinaryAccuracy.
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
    # if task == "binary":
    # Caution - now we are not using the AUROC metric here.
    # Adding it will cause this binary implementation to throw an error
    # due to the fact that AUROC does not have a multidim_average field.
    metric_function = metric_class(
        task=task,
        num_classes=num_classes,
        threshold=params["metrics"][metric_key]["threshold"],
        average=define_average_type_key(params, metric_key),
        multidim_average=define_multidim_average_type_key(params, metric_key),
    )
    # elif task == "multiclass":
    #     metric_function = metric_class(
    #         task=task,
    #         average=define_average_type_key(params, metric_key),
    #         num_classes=num_classes,
    #         threshold=params["metrics"][metric_key]["threshold"],
    #     )

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
        warnings.warn(
            f"WARNING! IoU score is not implemented for multilabel problems, setting recall to {recall}",
            UserWarning,
        )

    return generic_function_output_with_check(
        predicted_classes.cpu().int(), label.cpu().int(), recall
    )
