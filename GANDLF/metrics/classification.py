import torchmetrics as tm


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
    assert (
        params["problem_type"] == "classification"
    ), "Only classification is supported for overall stats"
    assert len(predictions) == len(
        ground_truth
    ), "Predictions and ground truth must be of same length"

    output_metrics = {}

    average_types_keys = {
        "global": "micro",
        "per_class": "none",
        "per_class_average": "macro",
        "per_class_weighted": "weighted",
    }
    # metrics that need the "average" parameter
    for average_type, average_type_key in average_types_keys.items():
        calculators = {
            "accuracy": tm.Accuracy(
                num_classes=params["model"]["num_classes"], average=average_type_key
            ),
            "precision": tm.Precision(
                num_classes=params["model"]["num_classes"], average=average_type_key
            ),
            "recall": tm.Recall(
                num_classes=params["model"]["num_classes"], average=average_type_key
            ),
            "f1": tm.F1(
                num_classes=params["model"]["num_classes"], average=average_type_key
            ),
            ## weird error for multi-class problem, where pos_label is not getting set
            # "aucroc": tm.AUROC(
            #     num_classes=params["model"]["num_classes"], average=average_type_key
            # ),
        }
        for metric_name, calculator in calculators.items():
            temp_output = calculator(predictions, ground_truth)
            if temp_output.dim() > 0:
                temp_output = temp_output.cpu().tolist()
            else:
                temp_output = temp_output.cpu().item()
            output_metrics[f"{metric_name}_{average_type}"] = temp_output
    # metrics that do not have any "average" parameter
    calculators = {
        "auc": tm.AUC(reorder=True),
        ## weird error for multi-class problem, where pos_label is not getting set
        # "roc": tm.ROC(num_classes=params["model"]["num_classes"]),
    }
    for metric_name, calculator in calculators.items():
        temp_output = calculator(predictions, ground_truth)
        if temp_output.dim() > 0:
            temp_output = temp_output.cpu().tolist()
        else:
            temp_output = temp_output.cpu().item()
        output_metrics[metric_name] = temp_output

    return output_metrics
