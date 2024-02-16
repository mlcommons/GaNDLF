import torch
import torchmetrics as tm
from torch.nn.functional import one_hot
from ..utils import get_output_from_calculator
from GANDLF.utils.generic import determine_classification_task_type


def overall_stats(prediction: torch.Tensor, target: torch.Tensor, params: dict) -> dict:
    """
    Generates a dictionary of metrics calculated on the overall prediction and ground truths.

    Args:
        prediction (torch.Tensor): The output of the model.
        target (torch.Tensor): The ground truth labels.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        dict: A dictionary of metrics.
    """
    assert (
        params["problem_type"] == "classification"
    ), "Only classification is supported for these stats"

    output_metrics = {}

    average_types_keys = {
        "global": "micro",
        "per_class": "none",
        "per_class_average": "macro",
        "per_class_weighted": "weighted",
    }
    task = determine_classification_task_type(params)
    # consider adding a "multilabel field in the future"
    # metrics that need the "average" parameter

    for average_type_key in average_types_keys.values():
        # multidim_average is not used when constructing these metrics
        # think of having it
        calculators = {
            "accuracy": tm.Accuracy(
                task=task,
                num_classes=params["model"]["num_classes"],
                average=average_type_key,
            ),
            "precision": tm.Precision(
                task=task,
                num_classes=params["model"]["num_classes"],
                average=average_type_key,
            ),
            "recall": tm.Recall(
                task=task,
                num_classes=params["model"]["num_classes"],
                average=average_type_key,
            ),
            "f1": tm.F1Score(
                task=task,
                num_classes=params["model"]["num_classes"],
                average=average_type_key,
            ),
            "specificity": tm.Specificity(
                task=task,
                num_classes=params["model"]["num_classes"],
                average=average_type_key,
            ),
            "aucroc": tm.AUROC(
                task=task,
                num_classes=params["model"]["num_classes"],
                average=average_type_key if average_type_key != "micro" else "macro",
            ),
        }
        for metric_name, calculator in calculators.items():
            if metric_name == "aucroc":
                one_hot_preds = one_hot(
                    prediction.long(),
                    num_classes=params["model"]["num_classes"],
                )
                output_metrics[metric_name] = get_output_from_calculator(
                    one_hot_preds.float(), target, calculator
                )
            else:
                output_metrics[metric_name] = get_output_from_calculator(
                    prediction, target, calculator
                )

    #### HERE WE NEED TO MODIFY TESTS - ROC IS RETURNING A TUPLE. WE MAY ALSO DISCARD IT ####
    # what is AUC metric telling at all? Computing it for prediction and ground truth
    # is not making sense
    # metrics that do not have any "average" parameter
    # calculators = {
    #
    #     # "auc": tm.AUC(reorder=True),
    #     ## weird error for multi-class problem, where pos_label is not getting set
    #     "roc": tm.ROC(task=task, num_classes=params["model"]["num_classes"]),
    # }
    # for metric_name, calculator in calculators.items():
    #     if metric_name == "roc":
    #         one_hot_preds = one_hot(
    #             prediction.long(), num_classes=params["model"]["num_classes"]
    #         )
    #         output_metrics[metric_name] = get_output_from_calculator(
    #             one_hot_preds.float(), target, calculator
    #         )
    #     else:
    #         output_metrics[metric_name] = get_output_from_calculator(
    #             prediction, target, calculator
    #         )

    return output_metrics
