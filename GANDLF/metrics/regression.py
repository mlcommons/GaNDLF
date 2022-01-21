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


def balanced_acc_score(output, label, params):
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    else:
        predicted_classes = output

    return torch.from_numpy(
        np.array(balanced_accuracy_score(predicted_classes.cpu(), label.cpu()))
    )
