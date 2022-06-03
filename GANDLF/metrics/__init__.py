"""
All the metrics are to be called from here
"""
from GANDLF.losses.regression import CEL, MSE_loss

from .generic import accuracy, f1_score, iou_score, precision_score, recall_score
from .regression import balanced_acc_score, classification_accuracy
from .segmentation import (
    hd95,
    hd95_per_label,
    hd100,
    hd100_per_label,
    multi_class_dice,
    multi_class_dice_per_label,
)

# global defines for the metrics
global_metrics_dict = {
    "dice": multi_class_dice,
    "dice_per_label": multi_class_dice_per_label,
    "accuracy": accuracy,
    "mse": MSE_loss,
    "hd95": hd95,
    "hd95_per_label": hd95_per_label,
    "hausdorff95": hd95,
    "hd100": hd100,
    "hd100_per_label": hd100_per_label,
    "hausdorff": hd100,
    "hausdorff100": hd100,
    "cel": CEL,
    "f1_score": f1_score,
    "f1": f1_score,
    "classification_accuracy": classification_accuracy,
    "precision": precision_score,
    "recall": recall_score,
    "iou": iou_score,
    "balanced_accuracy": balanced_acc_score,
}
