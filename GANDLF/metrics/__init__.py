"""
All the metrics are to be called from here
"""
from GANDLF.losses.regression import MSE_loss, CEL
from .segmentation import multi_class_dice, multi_class_dice_per_label, hd100, hd100_per_label, hd95, hd95_per_label
from .regression import accuracy, classification_accuracy, balanced_acc_score
from .generic import recall_score, precision_score, iou_score, f1_score


# global defines for the metrics
global_metrics_dict = {
    "dice": multi_class_dice,
    "dice_per_label": multi_class_dice_per_label,
    "accuracy": accuracy,
    "mse": MSE_loss,
    "hd95": hd95,
    "hd95_per_label": hd95_per_label,
    "hausdorff95": hd100,
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
