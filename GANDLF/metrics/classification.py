import torch
import torchmetrics as tm
import torch.nn.functional as F

# from torch.nn.functional import one_hot
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

    # this is needed for auroc
    # ensure that predictions are in integer format
    prediction_wrap = prediction
    if prediction.dtype != torch.long or prediction.dtype != torch.int:
        prediction_wrap = prediction_wrap.long()
    predictions_one_hot = F.one_hot(
        prediction_wrap, num_classes=params["model"]["num_classes"]
    )
    predictions_prob = F.softmax(predictions_one_hot.float(), dim=1)

    output_metrics = {}

    average_types_keys = {
        "global": "micro",
        "per_class": "none",
        "per_class_average": "macro",
        "per_class_weighted": "weighted",
    }
    task = determine_classification_task_type(params)
    # todo: consider adding a "multilabel field in the future"

    # metrics that need the "average" parameter
    for average_type, average_type_key in average_types_keys.items():
        # multidim_average is not used when constructing these metrics
        # think of having it
        calculators = {
            f"accuracy_{average_type}": tm.Accuracy(
                task=task,
                num_classes=params["model"]["num_classes"],
                average=average_type_key,
            ),
            f"precision_{average_type}": tm.Precision(
                task=task,
                num_classes=params["model"]["num_classes"],
                average=average_type_key,
            ),
            f"recall_{average_type}": tm.Recall(
                task=task,
                num_classes=params["model"]["num_classes"],
                average=average_type_key,
            ),
            f"f1_{average_type}": tm.F1Score(
                task=task,
                num_classes=params["model"]["num_classes"],
                average=average_type_key,
            ),
            f"specificity_{average_type}": tm.Specificity(
                task=task,
                num_classes=params["model"]["num_classes"],
                average=average_type_key,
            ),
            f"auroc_{average_type}": tm.AUROC(
                task=task,
                num_classes=params["model"]["num_classes"],
                average=average_type_key if average_type_key != "micro" else "macro",
            ),
        }
        for metric_name, calculator in calculators.items():
            if "auroc" in metric_name:
                output_metrics[metric_name] = get_output_from_calculator(
                    predictions_prob, target, calculator
                )
            else:
                output_metrics[metric_name] = get_output_from_calculator(
                    prediction, target, calculator
                )

    # metrics that do not need the "average" parameter
    calculators = {
        "mcc": tm.MatthewsCorrCoef(
            task=task, num_classes=params["model"]["num_classes"]
        )
    }
    for metric_name, calculator in calculators.items():
        output_metrics[metric_name] = get_output_from_calculator(
            prediction, target, calculator
        )

    return output_metrics
