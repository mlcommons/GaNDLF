import yaml
from pprint import pprint
import pandas as pd
import SimpleITK as sitk
import torch

from GANDLF.parseConfig import parseConfig
from GANDLF.utils import find_problem_type_from_parameters, one_hot
from GANDLF.metrics import overall_stats, multi_class_dice
from GANDLF.metrics.segmentation import (
    _calculator_generic_all_surface_distances,
    _calculator_sensitivity_specificity,
    _calculator_jaccard,
)

# input: 1 csv with 3 columns: subjectid, prediction, target
# input: gandlf config
# output: metrics.csv (based on the config)


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
                headers[column_to_check] = column_to_check
    for column in required_columns:
        assert column in headers, f"The input csv should have a column named {column}"

    parameters = parseConfig(config)
    problem_type = find_problem_type_from_parameters(parameters)
    parameters["problem_type"] = problem_type
    if problem_type == "regression" or problem_type == "classification":
        predictions_array = input_df[headers["prediction"]].to_numpy().ravel()
        labels_array = input_df[headers["target"]].to_numpy().ravel()
        overall_stats_dict = overall_stats(predictions_array, labels_array, parameters)
    elif problem_type == "segmentation":
        # read images and then calculate metrics
        class_list = parameters["model"]["class_list"]
        for _, row in input_df.iterrows():
            current_subject_id = row[headers["subject_id"]]
            overall_stats_dict[current_subject_id] = {}
            label_image = sitk.ReadImage(row["target"])
            pred_image = sitk.ReadImage(row["prediction"])
            label_tensor = torch.from_numpy(sitk.GetArrayFromImage(label_image))
            pred_tensor = torch.from_numpy(sitk.GetArrayFromImage(pred_image))

            # one hot encode with batch_size = 1
            label_image_one_hot = one_hot(label_tensor, class_list).unsqueeze(0)
            pred_image_one_hot = one_hot(pred_tensor, class_list).unsqueeze(0)

            for class_index, _ in enumerate(class_list):
                # this is inconsequential, since one_hot will ensure that the classes are present
                parameters["model"]["class_list"] = [0, 1]
                parameters["model"]["ignore_label_validation"] = 0
                overall_stats_dict[current_subject_id][
                    "dice_" + str(class_index)
                ] = multi_class_dice(
                    pred_image_one_hot,
                    label_image_one_hot,
                    parameters,
                ).item()
                overall_stats_dict[current_subject_id]["nsd_" + str(class_index)],
                overall_stats_dict[current_subject_id]["hd100_" + str(class_index)],
                overall_stats_dict[current_subject_id][
                    "hd95_" + str(class_index)
                ] = _calculator_generic_all_surface_distances(
                    pred_image_one_hot,
                    label_image_one_hot,
                    parameters,
                )
                (
                    overall_stats_dict[current_subject_id][
                        "sensitivity_" + str(class_index)
                    ],
                    overall_stats_dict[current_subject_id][
                        "specificity_" + str(class_index)
                    ],
                ) = _calculator_sensitivity_specificity(
                    pred_image_one_hot,
                    label_image_one_hot,
                    parameters,
                )
                overall_stats_dict[current_subject_id][
                    "jaccard_" + str(class_index)
                ] = _calculator_jaccard(
                    pred_image_one_hot,
                    label_image_one_hot,
                    parameters,
                )

    pprint(overall_stats_dict)
    if outputfile is not None:
        with open(outputfile, "w") as outfile:
            yaml.dump(overall_stats_dict, outfile)
