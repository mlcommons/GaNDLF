"""
All the metrics are to be called from here
"""
import torch
from torchmetrics import F1, Precision, Recall, IoU


def F1_score(output, label, params):
    num_classes = params["model"]["num_classes"]
    predicted_classes = torch.argmax(output, 1)
    f1 = F1(num_classes=num_classes)
    f1_score = f1(predicted_classes.cpu(), label.cpu())

    return f1_score


def classification_accuracy(output, label, params):
    acc = torch.sum(torch.argmax(output, 1) == label) / len(label)
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
    predicted_classes = torch.argmax(output, 1)
    precision = Precision(average=params["metrics"]["precision"]["average"], num_classes=num_classes)

    return precision(predicted_classes.cpu(), label.cpu())

def recall_score(output, label, params):
    num_classes = params["model"]["num_classes"]
    predicted_classes = torch.argmax(output, 1)
    recall = Recall(average=params["metrics"]["recall"]["average"], num_classes=num_classes)

    return recall(predicted_classes.cpu(), label.cpu())

def iou_score(output, label, params):
    num_classes = params["model"]["num_classes"]
    predicted_classes = torch.argmax(output, 1)
    recall = IoU(reduction=params["metrics"]["iou"]["reduction"], num_classes=num_classes, threshold=params["metrics"]["iou"]["threshold"])

    return recall(predicted_classes.cpu(), label.cpu())

