import yaml
from pprint import pprint
import pandas as pd
import torch
import torchio

from GANDLF.parseConfig import parseConfig
from GANDLF.utils import find_problem_type_from_parameters, one_hot
from GANDLF.metrics import overall_stats
from GANDLF.losses.segmentation import dice
from GANDLF.metrics.segmentation import (
    _calculator_generic_all_surface_distances,
    _calculator_sensitivity_specificity,
    _calculator_jaccard,
)


def generate_metrics_dict(input_csv: str, config: str, outputfile: str = None) -> dict:
    """
    This function generates metrics from the input csv and the config.

    Args:
        input_csv (str): The input CSV.
        config (str): The input yaml config.
        outputfile (str, optional): The output file to save the metrics. Defaults to None.

    Returns:
        dict: The metrics dictionary.
    """
    input_df = pd.read_csv(input_csv)

    # check required headers in a case insensitive manner
    headers = {}
    required_columns = ["subjectid", "prediction", "target"]
    for col, _ in input_df.iteritems():
        col_lower = col.lower()
        for column_to_check in required_columns:
            if column_to_check == col_lower:
                headers[column_to_check] = col
    for column in required_columns:
        assert column in headers, f"The input csv should have a column named {column}"

    overall_stats_dict = {}
    parameters = parseConfig(config)
    problem_type = find_problem_type_from_parameters(parameters)
    parameters["problem_type"] = problem_type

    if problem_type == "regression" or problem_type == "classification":
        parameters["model"]["num_classes"] = len(parameters["model"]["class_list"])
        predictions_tensor = torch.from_numpy(
            input_df[headers["prediction"]].to_numpy().ravel()
        )
        labels_tensor = torch.from_numpy(input_df[headers["target"]].to_numpy().ravel())
        overall_stats_dict = overall_stats(
            predictions_tensor, labels_tensor, parameters
        )
    elif problem_type == "segmentation":
        # read images and then calculate metrics
        class_list = parameters["model"]["class_list"]
        for _, row in input_df.iterrows():
            current_subject_id = row[headers["subjectid"]]
            overall_stats_dict[current_subject_id] = {}
            label_image = torchio.LabelMap(row["target"])
            pred_image = torchio.LabelMap(row["prediction"])
            label_tensor = label_image.data
            pred_tensor = pred_image.data
            spacing = label_image.spacing
            if label_tensor.data.shape[-1] == 1:
                spacing = spacing[0:2]
            # add dimension for batch
            parameters["subject_spacing"] = torch.Tensor(spacing).unsqueeze(0)
            label_tensor = label_tensor.unsqueeze(0)
            pred_tensor = pred_tensor.unsqueeze(0)

            # one hot encode with batch_size = 1
            label_image_one_hot = one_hot(label_tensor, class_list)
            pred_image_one_hot = one_hot(pred_tensor, class_list)

            for class_index, _ in enumerate(class_list):
                overall_stats_dict[current_subject_id][str(class_index)] = {}
                # this is inconsequential, since one_hot will ensure that the classes are present
                parameters["model"]["class_list"] = [0, 1]
                parameters["model"]["num_classes"] = 2
                parameters["model"]["ignore_label_validation"] = 0
                overall_stats_dict[current_subject_id][str(class_index)]["dice"] = dice(
                    pred_image_one_hot,
                    label_image_one_hot,
                ).item()
                nsd, hd100, hd95 = _calculator_generic_all_surface_distances(
                    pred_image_one_hot,
                    label_image_one_hot,
                    parameters,
                )
                overall_stats_dict[current_subject_id][str(class_index)][
                    "nsd"
                ] = nsd.item()
                overall_stats_dict[current_subject_id][str(class_index)][
                    "hd100"
                ] = hd100.item()
                overall_stats_dict[current_subject_id][str(class_index)][
                    "hd95"
                ] = hd95.item()

                (
                    s,
                    p,
                ) = _calculator_sensitivity_specificity(
                    pred_image_one_hot,
                    label_image_one_hot,
                    parameters,
                )
                overall_stats_dict[current_subject_id][str(class_index)][
                    "sensitivity"
                ] = s.item()
                overall_stats_dict[current_subject_id][str(class_index)][
                    "specificity"
                ] = p.item()
                overall_stats_dict[current_subject_id][
                    "jaccard_" + str(class_index)
                ] = _calculator_jaccard(
                    pred_image_one_hot,
                    label_image_one_hot,
                    parameters,
                ).item()

    pprint(overall_stats_dict)
    if outputfile is not None:
        with open(outputfile, "w") as outfile:
            yaml.dump(overall_stats_dict, outfile)
