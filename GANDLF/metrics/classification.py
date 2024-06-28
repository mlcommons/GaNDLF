import torch
import torchmetrics as tm
from sklearn.metrics import matthews_corrcoef

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
            "auroc": tm.AUROC(
                task=task,
                num_classes=params["model"]["num_classes"],
                average=average_type_key if average_type_key != "micro" else "macro",
            ),
        }
        for metric_name, calculator in calculators.items():
            # TODO: AUROC needs to be properly debugged for multi-class problems - https://github.com/mlcommons/GaNDLF/issues/817
            if metric_name == "auroc" and params["model"]["num_classes"] == 2:
                output_metrics[metric_name] = get_output_from_calculator(
                    prediction, target, calculator
                )
            elif metric_name != "auroc":
                output_metrics[metric_name] = get_output_from_calculator(
                    prediction, target, calculator
                )

    output_metrics["mcc"] = matthews_corrcoef(
        target.cpu().numpy(), prediction.cpu().numpy()
    )

    return output_metrics
