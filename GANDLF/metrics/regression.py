"""
All the metrics are to be called from here
"""
import torch
from sklearn.metrics import balanced_accuracy_score
import numpy as np


def classification_accuracy(output, label, params):
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    else:
        predicted_classes = output
    acc = torch.sum(predicted_classes == label) / len(label)
    return acc


def accuracy(output, label, params):
    """
    Calculates the accuracy between output and a label

    Parameters
    ----------
    output : torch.Tensor
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    label : torch.Tensor
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    params : dict
        The parameter dictionary containing training and data information.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if params["metrics"]["accuracy"]["threshold"] is not None:
        output = (output >= params["metrics"]["accuracy"]["threshold"]).float()
    correct = (output == label).float().sum()
    return correct / len(label)


def balanced_acc_score(output, label, params):
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    else:
        predicted_classes = output

    return torch.from_numpy(
        np.array(balanced_accuracy_score(predicted_classes.cpu(), label.cpu()))
    )
