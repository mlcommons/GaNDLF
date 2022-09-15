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
        for _output, _label in zip(output, label):
            predicted_classes = torch.Tensor([0] * len(params["model"]["class_list"]))
            label_cpu = torch.Tensor([0] * len(params["model"]["class_list"]))
            predicted_classes[torch.argmax(_output, 0).cpu().item()] = 1
            label_cpu[_label.cpu().item()] = 1
            output_accuracy += (predicted_classes == label_cpu).type(torch.float)
        return output_accuracy / len(output)
    else:
        return balanced_acc_score(output, label, params)
