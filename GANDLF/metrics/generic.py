import torch
from torchmetrics import F1, Precision, Recall, IoU
from GANDLF.utils.tensor import one_hot


def generic_torchmetrics_score(output, label, metric_class, metric_key, params):
    num_classes = params["model"]["num_classes"]
    predicted_classes = output
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    elif params["problem_type"] == "segmentation":
        label = one_hot(label, params["model"]["class_list"])
    else:
        params["metrics"][metric_key]["multi_class"] = False
        params["metrics"][metric_key]["mdmc_average"] = None
    metric_function = metric_class(average=params["metrics"][metric_key]["average"], num_classes=num_classes, multiclass=params["metrics"][metric_key]["multi_class"],mdmc_average=params["metrics"][metric_key]["mdmc_average"], threshold=params["metrics"][metric_key]["threshold"])

    return metric_function(predicted_classes.cpu(), label.cpu())

def recall_score(output, label, params):
    return generic_torchmetrics_score(output, label, Recall, "recall", params)


def precision_score(output, label, params):
    return generic_torchmetrics_score(output, label, Precision, "precision", params)


def f1_score(output, label, params):
    return generic_torchmetrics_score(output, label, F1, "f1", params)


def iou_score(output, label, params):
    num_classes = params["model"]["num_classes"]
    predicted_classes = output
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    elif params["problem_type"] == "segmentation":
        label = one_hot(label, params["model"]["class_list"])
    
    recall = IoU(reduction=params["metrics"]["iou"]["reduction"], num_classes=num_classes, threshold=params["metrics"]["iou"]["threshold"])

    return recall(predicted_classes.cpu(), label.cpu())
