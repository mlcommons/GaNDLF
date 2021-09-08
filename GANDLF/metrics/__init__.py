"""
All the metrics are to be called from here
"""
from GANDLF.losses.regression import MSE_loss, CEL
from .segmentation import multi_class_dice, hd100, hd95
from .regression import accuracy, F1_score, classification_accuracy, balanced_acc_score
from .generic import recall_score, precision_score, iou_score


# global defines for the metrics
global_metrics_dict = {
    "dice": multi_class_dice,
    "accuracy": accuracy,
    "mse": MSE_loss,
    "hd95": hd95,
    "hausdorff95": hd100,
    "hd100": hd100,
    "hausdorff": hd100,
    "hausdorff100": hd100,
    "cel": CEL,
    "f1_score": F1_score,
    "f1": F1_score,
    "classification_accuracy": classification_accuracy,
    "precision": precision_score,
    "recall": recall_score,
    "iou": iou_score,
    "balanced_accuracy": balanced_acc_score,
}
