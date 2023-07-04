"""
All the metrics are to be called from here
"""
import torch
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import torchmetrics as tm
from ..utils import get_output_from_calculator


def classification_accuracy(output, label, params):
    """
    This function computes the classification accuracy.

    Args:
        output (torch.Tensor): The output of the model.
        label (torch.Tensor): The ground truth labels.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        torch.Tensor: The classification accuracy.
    """
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    else:
        predicted_classes = output

    acc = torch.sum(predicted_classes == label.squeeze()) / len(label)
    return acc


def balanced_acc_score(output, label, params):
    """
    This function computes the balanced accuracy.

    Args:
        output (torch.Tensor): The output of the model.
        label (torch.Tensor): The ground truth labels.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        torch.Tensor: The balanced accuracy.
    """
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    else:
        predicted_classes = output

    return torch.from_numpy(
        np.array(balanced_accuracy_score(predicted_classes.cpu(), label.cpu()))
    )


def per_label_accuracy(output, label, params):
    """
    This function computes the per class accuracy.

    Args:
        output (torch.Tensor): The output of the model.
        label (torch.Tensor): The ground truth labels.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        torch.Tensor: The per class accuracy.
    """
    if params["problem_type"] == "classification":
        # ensure this works for multiple batches
        output_accuracy = torch.zeros(len(params["model"]["class_list"]))
        for output_batch, label_batch in zip(output, label):
            predicted_classes = torch.Tensor([0] * len(params["model"]["class_list"]))
            label_cpu = torch.Tensor([0] * len(params["model"]["class_list"]))
            predicted_classes[torch.argmax(output_batch, 0).cpu().item()] = 1
            label_cpu[label_batch.cpu().item()] = 1
            output_accuracy += (predicted_classes == label_cpu).type(torch.float)
        return output_accuracy / len(output)
    else:
        return balanced_acc_score(output, label, params)


def overall_stats(predictions, ground_truth, params):
    """
    Generates a dictionary of metrics calculated on the overall predictions and ground truths.

    Args:
        predictions (torch.Tensor): The output of the model.
        ground_truth (torch.Tensor): The ground truth labels.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        dict: A dictionary of metrics.
    """
    predictions = predictions.type(torch.float)
    ground_truth = ground_truth.type(torch.float) * params["scaling_factor"]
    assert (
        params["problem_type"] == "regression"
    ), "Only regression is supported for these stats"

    output_metrics = {}

    reduction_types_keys = {
        "mean": "mean",
        "sum": "sum",
        "none": "none",
    }
    # metrics that need the "reduction" parameter
    for reduction_type, reduction_type_key in reduction_types_keys.items():
        calculators = {
            "cosinesimilarity": tm.CosineSimilarity(reduction=reduction_type_key),
        }
        for metric_name, calculator in calculators.items():
            output_metrics[
                f"{metric_name}_{reduction_type}"
            ] = get_output_from_calculator(predictions, ground_truth, calculator)
    # metrics that do not have any "reduction" parameter
    calculators = {
        "mse": tm.MeanSquaredError(),
        "mae": tm.MeanAbsoluteError(),
        "pearson": tm.PearsonCorrCoef(),
        "spearman": tm.SpearmanCorrCoef(),
    }
    for metric_name, calculator in calculators.items():
        output_metrics[metric_name] = get_output_from_calculator(
            predictions, ground_truth, calculator
        )

    return output_metrics
