"""
All the metrics are to be called from here
"""
import torch
from sklearn.metrics import balanced_accuracy_score
import numpy as np


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


def per_class_accuracy(output, label, params):
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
        # acc = [0 for _ in params["model"]["class_list"]]
        predicted_classes = np.array([0 for _ in params["model"]["class_list"]])
        label_cpu = np.array([0 for _ in params["model"]["class_list"]])
        predicted_classes[torch.argmax(output, 1).cpu().item()] = 1
        label_cpu[label.cpu().item()] = 1
        return torch.from_numpy((predicted_classes == label_cpu).astype(float))
    else:
        return balanced_acc_score(output, label, params)
