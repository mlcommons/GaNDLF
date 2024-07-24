"""
All the metrics are to be called from here
"""
from typing import Union

import torch
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import torchmetrics as tm
from ..utils import get_output_from_calculator


def classification_accuracy(
    prediction: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    """
    This function computes the classification accuracy.

    Args:
        prediction (torch.Tensor): The prediction of the model.
        target (torch.Tensor): The ground truth labels.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        torch.Tensor: The classification accuracy.
    """
    predicted_classes = prediction
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(prediction, 1)

    acc = torch.sum(predicted_classes == target.squeeze()) / len(target)
    return acc


def balanced_acc_score(
    prediction: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    """
    This function computes the balanced accuracy.

    Args:
        prediction (torch.Tensor): The prediction of the model.
        target (torch.Tensor): The ground truth labels.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        torch.Tensor: The balanced accuracy.
    """
    predicted_classes = prediction
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(prediction, 1)

    return torch.from_numpy(
        np.array(balanced_accuracy_score(predicted_classes.cpu(), target.cpu()))
    )


def per_label_accuracy(
    prediction: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    """
    This function computes the per class accuracy.

    Args:
        prediction (torch.Tensor): The prediction of the model.
        target (torch.Tensor): The ground truth labels.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        torch.Tensor: The per class accuracy.
    """
    if params["problem_type"] == "classification":
        # ensure this works for multiple batches
        output_accuracy = torch.zeros(len(params["model"]["class_list"]))
        for output_batch, label_batch in zip(prediction, target):
            predicted_classes = torch.Tensor([0] * len(params["model"]["class_list"]))
            label_cpu = torch.Tensor([0] * len(params["model"]["class_list"]))
            predicted_classes[torch.argmax(output_batch, 0).cpu().item()] = 1
            label_cpu[label_batch.cpu().item()] = 1
            output_accuracy += (predicted_classes == label_cpu).type(torch.float)
        return output_accuracy / len(prediction)
    else:
        return balanced_acc_score(prediction, target, params)


def overall_stats(
    prediction: torch.Tensor, target: torch.Tensor, params: dict
) -> dict[str, Union[float, list]]:
    """
    Generates a dictionary of metrics calculated on the overall predictions and ground truths.

    Args:
        predictions (torch.Tensor): The prediction of the model.
        target (torch.Tensor): The ground truth labels.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        dict: A dictionary of metrics.
    """
    prediction = prediction.type(torch.float)
    target = target.type(torch.float) * params["scaling_factor"]
    assert (
        params["problem_type"] == "regression"
    ), "Only regression is supported for these stats"

    output_metrics = {}

    reduction_types_keys = {"mean": "mean", "sum": "sum", "none": "none"}
    # metrics that need the "reduction" parameter
    for reduction_type, reduction_type_key in reduction_types_keys.items():
        calculators = {
            "cosinesimilarity": tm.CosineSimilarity(reduction=reduction_type_key)
        }
        for metric_name, calculator in calculators.items():
            output_metrics[
                f"{metric_name}_{reduction_type}"
            ] = get_output_from_calculator(prediction, target, calculator)
    # metrics that do not have any "reduction" parameter
    calculators = {
        "mse": tm.MeanSquaredError(),
        "mae": tm.MeanAbsoluteError(),
        "pearson": tm.PearsonCorrCoef(),
        "spearman": tm.SpearmanCorrCoef(),
    }
    for metric_name, calculator in calculators.items():
        output_metrics[metric_name] = get_output_from_calculator(
            prediction, target, calculator
        )

    return output_metrics
