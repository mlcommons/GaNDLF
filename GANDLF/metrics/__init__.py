"""
All the metrics are to be called from here
"""
from GANDLF.losses.regression import MSE_loss, CEL
from .segmentation import (
    multi_class_dice,
    multi_class_dice_per_label,
    hd100,
    hd100_per_label,
    hd95,
    hd95_per_label,
)
from .regression import classification_accuracy, balanced_acc_score, per_label_accuracy
from .generic import recall_score, precision_score, iou_score, f1_score, accuracy
import GANDLF.metrics.classification as classification
import GANDLF.metrics.regression as regression


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
    "per_label_one_hot_accuracy": per_label_accuracy,
}


def overall_stats(predictions, ground_truth, params):
    """
    Generates a dictionary of metrics calculated on the overall predictions and ground truths.

    Args:
        predictions (torch.Tensor): The output of the model.
        ground_truth (torch.Tensor): The ground truth labels.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        dict: A dictionary of metrics.
    """
    assert len(predictions) == len(
        ground_truth
    ), "Predictions and ground truth must be of same length"

    if params["problem_type"] == "classification":
        return classification.overall_stats(predictions, ground_truth, params)
    elif params["problem_type"] == "regression":
        return regression.overall_stats(predictions, ground_truth, params)
