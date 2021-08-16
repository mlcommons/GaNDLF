"""
All the metrics are to be called from here
"""
import torch
from GANDLF.losses import MSE_loss, cel
from .segmentation import multi_class_dice, hd100, hd95
from .regression import accuracy, F1_score, classification_accuracy


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
    "cel": cel,
    "f1_score": F1_score,
    "f1": F1_score,
    "classification_accuracy": classification_accuracy,
}