import sys
import json
from typing import Optional
from pprint import pprint
import pandas as pd
from tqdm import tqdm
import torch
import torchio
import SimpleITK as sitk
import numpy as np

from GANDLF.config_manager import ConfigManager
from GANDLF.utils import find_problem_type_from_parameters, one_hot
from GANDLF.metrics import (
    overall_stats,
    structural_similarity_index,
    mean_squared_error,
    root_mean_squared_error,
    peak_signal_noise_ratio,
    mean_squared_log_error,
    mean_absolute_error,
    ncc_metrics,
    generate_instance_segmentation,
)
from GANDLF.losses.segmentation import dice
from GANDLF.metrics.segmentation import (
    _calculator_generic_all_surface_distances,
    _calculator_sensitivity_specificity,
    _calculator_jaccard,
)


def __update_header_location_case_insensitive(
    input_df: pd.DataFrame, expected_column_name: str, required: bool = True
) -> pd.DataFrame:
    """
    This function checks for a column in the dataframe in a case-insensitive manner and renames it.

    Args:
        input_df (pd.DataFrame): The input dataframe.
        expected_column_name (str): The expected column name.
        required (bool, optional): Whether the column is required. Defaults to True.

    Returns:
        pd.DataFrame: The updated dataframe.
    """
    actual_column_name = None
    for col in input_df.columns:
        if col.lower() == expected_column_name.lower():
            actual_column_name = col
            break

    if required:
        assert (
            actual_column_name is not None
        ), f"Column {expected_column_name} not found in the dataframe"

        return input_df.rename(columns={actual_column_name: expected_column_name})
    else:
        return input_df


def generate_metrics_dict(
    input_csv: str,
    config: str,
    outputfile: Optional[str] = None,
    missing_prediction: int = -1,
) -> dict:
    """
    This function generates metrics from the input csv and the config.

    Args:
        input_csv (str): The input CSV.
        config (str): The input yaml config.
        outputfile (str, optional): The output file to save the metrics. Defaults to None.
        missing_prediction (int, optional): The value to use for missing predictions as penalty. Default is -1.

    Returns:
        dict: The metrics dictionary.
    """
    # the case where the input is a comma-separated 2 files with targets and predictions
    if "," in input_csv:
        target_csv, prediction_csv = input_csv.split(",")
        target_df = pd.read_csv(target_csv)
        prediction_df = pd.read_csv(prediction_csv)
        ## start sanity checks
        # if missing predictions are not to be penalized, check if the number of rows in the target and prediction files are the same
        if missing_prediction == -1:
            assert (
                target_df.shape[0] == prediction_df.shape[0]
            ), "The number of rows in the target and prediction files should be the same"

        # check if the number of columns in the target and prediction files are the same
        assert (
            target_df.shape[1] == prediction_df.shape[1]
        ), "The number of columns in the target and prediction files should be the same"
        assert (
            target_df.shape[1] == 2
        ), "The target and prediction files should have *exactly* 2 columns"

        # find the correct header for the subjectID column
        target_df = __update_header_location_case_insensitive(target_df, "SubjectID")
        prediction_df = __update_header_location_case_insensitive(
            prediction_df, "SubjectID"
        )
        # check if prediction_df has extra subjectIDs
        assert (
            prediction_df["SubjectID"].isin(target_df["SubjectID"]).all()
        ), "The `SubjectID` column in the prediction file should be a subset of the `SubjectID` column in the target file"

        # individual checks for target and prediction dataframes
        for df in [target_df, prediction_df]:
            # check if the "subjectID" column has duplicates
            assert (
                df["SubjectID"].duplicated().sum() == 0
            ), "The `SubjectID` column should not have duplicates"

            # check if SubjectID is the first column
            assert (
                df.columns[0] == "SubjectID"
            ), "The `SubjectID` column should be the first column in the target and prediction files"

        # change the column name after subjectID to target and prediction
        target_df = target_df.rename(columns={target_df.columns[1]: "Target"})
        prediction_df = prediction_df.rename(
            columns={prediction_df.columns[1]: "Prediction"}
        )

        # combine the two dataframes
        input_df = target_df.merge(prediction_df, how="left", on="SubjectID").fillna(
            missing_prediction
        )

    else:
        # the case where the input is a single file with targets and predictions
        input_df = pd.read_csv(input_csv)

        # check required headers in a case insensitive manner and rename them
        required_columns = ["SubjectID", "Prediction", "Target"]
        for column_to_check in required_columns:
            input_df = __update_header_location_case_insensitive(
                input_df, column_to_check
            )

        # check if the "subjectID" column has duplicates
        assert (
            input_df["SubjectID"].duplicated().sum() == 0
        ), "The `SubjectID` column should not have duplicates"

    overall_stats_dict = {}
    parameters = ConfigManager(config)
    # ensure that the problem_type is set
    problem_type = parameters.get("problem_type", None)
    problem_type = (
        find_problem_type_from_parameters(parameters)
        if problem_type is None
        else problem_type
    )
    parameters["problem_type"] = problem_type

    if problem_type == "classification":
        parameters["model"]["num_classes"] = parameters["model"].get(
            "num_classes", len(parameters["model"]["class_list"])
        )

    if problem_type == "regression" or problem_type == "classification":
        predictions_tensor = torch.from_numpy(input_df["Prediction"].to_numpy().ravel())
        labels_tensor = torch.from_numpy(input_df["Target"].to_numpy().ravel())
        overall_stats_dict = overall_stats(
            predictions_tensor, labels_tensor, parameters
        )

    elif problem_type == "segmentation":
        # read images and then calculate metrics
        class_list = parameters["model"]["class_list"]
        for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
            current_subject_id = row["SubjectID"]
            overall_stats_dict[current_subject_id] = {}
            label_image = torchio.LabelMap(row["Target"])
            pred_image = torchio.LabelMap(row["Prediction"])
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
                current_target = label_image_one_hot[:, class_index, ...].unsqueeze(0)
                current_prediction = pred_image_one_hot[:, class_index, ...].unsqueeze(
                    0
                )
                overall_stats_dict[current_subject_id][str(class_index)] = {}
                # this is inconsequential, since one_hot will ensure that the classes are present
                parameters["model"]["class_list"] = [1]
                parameters["model"]["num_classes"] = 1
                overall_stats_dict[current_subject_id][str(class_index)]["dice"] = dice(
                    current_prediction, current_target
                ).item()
                nsd, hd100, hd95 = _calculator_generic_all_surface_distances(
                    current_prediction, current_target, parameters
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

                (s, p) = _calculator_sensitivity_specificity(
                    current_prediction, current_target, parameters
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
                    current_prediction, current_target, parameters
                ).item()
                current_target_image = sitk.GetImageFromArray(
                    current_target[0, 0, ...].long()
                )
                current_prediction_image = sitk.GetImageFromArray(
                    current_prediction[0, 0, ...].long()
                )
                label_overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
                label_overlap_filter.Execute(
                    current_target_image, current_prediction_image
                )
                # overall_stats_dict[current_subject_id][
                #     "falseDiscoveryRate_" + str(class_index)
                # ] = label_overlap_filter.GetFalseDiscoveryRate()
                overall_stats_dict[current_subject_id][
                    "falseNegativeError_" + str(class_index)
                ] = label_overlap_filter.GetFalseNegativeError()
                overall_stats_dict[current_subject_id][
                    "falsePositiveError_" + str(class_index)
                ] = label_overlap_filter.GetFalsePositiveError()
                overall_stats_dict[current_subject_id][
                    "meanOverlap_" + str(class_index)
                ] = label_overlap_filter.GetMeanOverlap()
                overall_stats_dict[current_subject_id][
                    "unionOverlap_" + str(class_index)
                ] = label_overlap_filter.GetUnionOverlap()
                overall_stats_dict[current_subject_id][
                    "volumeSimilarity_" + str(class_index)
                ] = label_overlap_filter.GetVolumeSimilarity()

    elif problem_type == "segmentation_brats":
        for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
            current_subject_id = row["SubjectID"]
            overall_stats_dict[current_subject_id] = {}
            label_image = torchio.LabelMap(row["Target"])
            pred_image = torchio.LabelMap(row["Prediction"])
            label_tensor = label_image.data
            pred_tensor = pred_image.data
            spacing = label_image.spacing
            if label_tensor.data.shape[-1] == 1:
                spacing = spacing[0:2]
            # add dimension for batch
            parameters["subject_spacing"] = torch.Tensor(spacing).unsqueeze(0)
            label_array = label_tensor.unsqueeze(0).numpy()
            pred_array = pred_tensor.unsqueeze(0).numpy()

            overall_stats_dict[current_subject_id] = generate_instance_segmentation(
                prediction=pred_array, target=label_array
            )

    elif problem_type == "synthesis":

        def __fix_2d_tensor(input_tensor):
            """
            This function checks for 2d images and change the shape to [B, C, H, W]

            Args:
                input_tensor (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor in the format that torchmetrics expects.
            """
            if input_tensor.shape[-1] == 1:
                return input_tensor.squeeze(-1).unsqueeze(0)
            else:
                return input_tensor

        def __percentile_clip(
            input_tensor: torch.Tensor,
            reference_tensor: torch.Tensor = None,
            p_min: Optional[float] = 0.5,
            p_max: Optional[float] = 99.5,
            strictlyPositive: Optional[bool] = True,
        ):
            """
            Normalizes a tensor based on percentiles. Clips values below and above the percentile.
            Percentiles for normalization can come from another tensor.

            Args:
                input_tensor (torch.Tensor): Tensor to be normalized based on the data from the reference_tensor. If reference_tensor is None, the percentiles from this tensor will be used.
                reference_tensor (torch.Tensor, optional): The tensor used for obtaining the percentiles.
                p_min (float, optional): Lower end percentile. Defaults to 0.5.
                p_max (float, optional): Upper end percentile. Defaults to 99.5.
                strictlyPositive (bool, optional): Ensures that really all values are above 0 before normalization. Defaults to True.

            Returns:
                torch.Tensor: The input_tensor normalized based on the percentiles of the reference tensor.
            """
            reference_tensor = (
                input_tensor if reference_tensor is None else reference_tensor
            )
            # get p_min percentile and p_max percentile
            v_min, v_max = np.percentile(reference_tensor, [p_min, p_max])
            # set lower bound to be 0 if strictlyPositive is enabled
            v_min = max(v_min, 0.0) if strictlyPositive else v_min
            # clip values to percentiles from reference_tensor
            output_tensor = np.clip(input_tensor, v_min, v_max)
            # normalizes values to [0;1]
            output_tensor = (output_tensor - v_min) / (v_max - v_min)
            return output_tensor

        # these are additional columns that could be present for synthesis tasks
        for column_to_make_case_insensitive in ["Mask", "VoidImage"]:
            input_df = __update_header_location_case_insensitive(
                input_df, column_to_make_case_insensitive, False
            )

        for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
            current_subject_id = row["SubjectID"]
            overall_stats_dict[current_subject_id] = {}
            target_image = __fix_2d_tensor(torchio.ScalarImage(row["Target"]).data)
            pred_image = __fix_2d_tensor(torchio.ScalarImage(row["Prediction"]).data)
            # if "Mask" is not in the row, we assume that the whole image is the mask
            # always cast to byte tensor
            mask = (
                __fix_2d_tensor(torchio.LabelMap(row["Mask"]).data)
                if "Mask" in row
                else torch.from_numpy(
                    np.ones(target_image.numpy().shape, dtype=np.uint8)
                )
            ).byte()

            void_image_present = True if "VoidImage" in row else False
            void_image = (
                __fix_2d_tensor(torchio.ScalarImage(row["VoidImage"]).data)
                if "VoidImage" in row
                else torch.from_numpy(
                    np.ones(target_image.numpy().shape, dtype=np.uint8)
                )
            )

            # Get Infill region (we really are only interested in the infill region)
            output_infill = (pred_image * mask).float()
            gt_image_infill = (target_image * mask).float()

            # Normalize to [0;1] based on GT (otherwise MSE will depend on the image intensity range)
            normalize = parameters.get("normalize", True)
            if normalize:
                # use all the tissue that is not masked for normalization
                reference_tensor = (
                    target_image * ~mask if not void_image_present else void_image
                )
                gt_image_infill = __percentile_clip(
                    gt_image_infill,
                    reference_tensor=reference_tensor,
                    p_min=0.5,
                    p_max=99.5,
                    strictlyPositive=True,
                )
                output_infill = __percentile_clip(
                    output_infill,
                    reference_tensor=reference_tensor,
                    p_min=0.5,
                    p_max=99.5,
                    strictlyPositive=True,
                )

            overall_stats_dict[current_subject_id][
                "ssim"
            ] = structural_similarity_index(output_infill, gt_image_infill, mask).item()

            # ncc metrics
            compute_ncc = parameters.get("compute_ncc", True)
            if compute_ncc:
                calculated_ncc_metrics = ncc_metrics(output_infill, gt_image_infill)
                for key, value in calculated_ncc_metrics.items():
                    # we don't need the ".item()" here, since the values are already scalars
                    overall_stats_dict[current_subject_id][key] = value

            # only voxels that are to be inferred (-> flat array)
            # these are required for mse, psnr, etc.
            gt_image_infill = gt_image_infill[mask]
            output_infill = output_infill[mask]

            overall_stats_dict[current_subject_id]["mse"] = mean_squared_error(
                output_infill, gt_image_infill
            ).item()

            overall_stats_dict[current_subject_id]["rmse"] = root_mean_squared_error(
                output_infill, gt_image_infill
            ).item()

            overall_stats_dict[current_subject_id]["msle"] = mean_squared_log_error(
                output_infill, gt_image_infill
            ).item()

            overall_stats_dict[current_subject_id]["mae"] = mean_absolute_error(
                output_infill, gt_image_infill
            ).item()

            # torchmetrics PSNR using "max"
            overall_stats_dict[current_subject_id]["psnr"] = peak_signal_noise_ratio(
                output_infill, gt_image_infill
            ).item()

            # same as above but with epsilon for robustness
            overall_stats_dict[current_subject_id][
                "psnr_eps"
            ] = peak_signal_noise_ratio(
                output_infill, gt_image_infill, epsilon=sys.float_info.epsilon
            ).item()

            # only use fix data range to [0;1] if the data was normalized before
            if normalize:
                # torchmetrics PSNR but with fixed data range of 0 to 1
                overall_stats_dict[current_subject_id][
                    "psnr_01"
                ] = peak_signal_noise_ratio(
                    output_infill, gt_image_infill, data_range=(0, 1)
                ).item()

                # same as above but with epsilon for robustness
                overall_stats_dict[current_subject_id][
                    "psnr_01_eps"
                ] = peak_signal_noise_ratio(
                    output_infill,
                    gt_image_infill,
                    data_range=(0, 1),
                    epsilon=sys.float_info.epsilon,
                ).item()

    pprint(overall_stats_dict)
    if outputfile is not None:
        ## todo: needs debugging since this writes the file handler in some cases, so replaced with json
        # with open(outputfile, "w") as outfile:
        #     yaml.dump(overall_stats_dict, outfile)
        with open(outputfile, "w") as file:
            file.write(json.dumps(overall_stats_dict))
