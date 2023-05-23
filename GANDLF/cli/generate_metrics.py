import pandas as pd
import SimpleITK as sitk
import torch

from GANDLF.parseConfig import parseConfig
from GANDLF.utils import find_problem_type_from_parameters, one_hot
from GANDLF.metrics import overall_stats, multi_class_dice
from GANDLF.metrics.segmentation import _calculator_generic_all_surface_distances

# input: 1 csv with 3 columns: subjectid, prediction, ground_truth
# input: gandlf config
# output: metrics.csv (based on the config)


def generate_metrics_dict(input_csv: str, config: str) -> dict:
    """
    This function generates metrics from the input csv and the config.

    Args:
        input_csv (str): The input CSV.
        config (str): The input yaml config.

    Returns:
        dict: The metrics dictionary.
    """
    input_df = pd.read_csv(input_csv)

    required_columns = ["subjectid", "prediction", "ground_truth"]
    for column in required_columns:
        assert (
            column in input_df.columns
        ), f"The input csv should have a column named {column}"

    parameters = parseConfig(config)
    problem_type = find_problem_type_from_parameters(parameters)
    parameters["problem_type"] = problem_type
    if problem_type == "regression" or problem_type == "classification":
        predictions_array = input_df["prediction"].to_numpy().ravel()
        labels_array = input_df["ground_truth"].to_numpy().ravel()
        overall_stats_dict = overall_stats(predictions_array, labels_array, parameters)
    elif problem_type == "segmentation":
        # read images and then calculate metrics
        class_list = parameters["model"]["class_list"]
        for _, row in input_df.iterrows():
            current_subject_metrics = {}
            label_image = sitk.ReadImage(row["ground_truth"])
            pred_image = sitk.ReadImage(row["prediction"])
            label_tensor = torch.from_numpy(sitk.GetArrayFromImage(label_image))
            pred_tensor = torch.from_numpy(sitk.GetArrayFromImage(pred_image))

            # one hot encode with batch_size = 1
            label_image_one_hot = one_hot(label_tensor, class_list).unsqueeze(0)
            pred_image_one_hot = one_hot(pred_tensor, class_list).unsqueeze(0)

            for class_index, _ in enumerate(class_list):
                # this is inconsequential, since one_hot will ensure that the classes are present
                parameters["model"]["class_list"] = [0]
                current_subject_metrics["dice_" + str(class_index)] = multi_class_dice(
                    pred_image_one_hot,
                    label_image_one_hot,
                    parameters,
                ).item()
                current_subject_metrics["nsd_" + str(class_index)],
                current_subject_metrics["hd100_" + str(class_index)],
                current_subject_metrics[
                    "hd95_" + str(class_index)
                ] = _calculator_generic_all_surface_distances(
                    pred_image_one_hot,
                    label_image_one_hot,
                    parameters,
                )
