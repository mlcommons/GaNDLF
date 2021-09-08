import torch
from torchmetrics import F1, Precision, Recall, IoU
from GANDLF.utils.tensor import one_hot

def recall_score(output, label, params):
    num_classes = params["model"]["num_classes"]
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    elif params["problem_type"] == "segmentation":
        label = one_hot(label, params["model"]["class_list"])
        predicted_classes = output
    else:
        predicted_classes = output
    recall = Recall(average=params["metrics"]["recall"]["average"], num_classes=num_classes, multiclass=params["metrics"]["recall"]["multi_class"],mdmc_average="samplewise")

    return recall(predicted_classes.cpu(), label.cpu())
