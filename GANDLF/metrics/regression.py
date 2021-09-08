"""
All the metrics are to be called from here
"""
import torch
from torchmetrics import F1, Precision, Recall, IoU
from sklearn.metrics import balanced_accuracy_score
import numpy as np


def F1_score(output, label, params):
    num_classes = params["model"]["num_classes"]
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    else:
        predicted_classes = output
    f1 = F1(num_classes=num_classes)
    f1_score = f1(predicted_classes.cpu(), label.cpu())

    return f1_score


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

def precision_score(output, label, params):
    num_classes = params["model"]["num_classes"]
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    else:
        predicted_classes = output
    precision = Precision(average=params["metrics"]["precision"]["average"], num_classes=num_classes)

    return precision(predicted_classes.cpu(), label.cpu())

def iou_score(output, label, params):
    num_classes = params["model"]["num_classes"]
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    else:
        predicted_classes = output
    recall = IoU(reduction=params["metrics"]["iou"]["reduction"], num_classes=num_classes, threshold=params["metrics"]["iou"]["threshold"])

    return recall(predicted_classes.cpu(), label.cpu())

def balanced_acc_score(output, label, params):
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    else:
        predicted_classes = output

    return torch.from_numpy(np.array(balanced_accuracy_score(predicted_classes.cpu(), label.cpu())))

