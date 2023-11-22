import sys
import yaml
from pprint import pprint
import pandas as pd
from tqdm import tqdm
import torch
import torchio
import SimpleITK as sitk
import numpy as np

from GANDLF.parseConfig import parseConfig
from GANDLF.utils import find_problem_type_from_parameters, one_hot
from GANDLF.metrics import (
    overall_stats,
    structural_similarity_index,
    mean_squared_error,
    peak_signal_noise_ratio,
    mean_squared_log_error,
    mean_absolute_error,
    ncc_mean,
    ncc_std,
    ncc_max,
    ncc_min,
)
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
    for col, _ in input_df.items():
        col_lower = col.lower()
        for column_to_check in required_columns:
            if column_to_check == col_lower:
                headers[column_to_check] = col
        if col_lower == "mask":
            headers["mask"] = col
    for column in required_columns:
        assert column in headers, f"The input csv should have a column named {column}"

    overall_stats_dict = {}
    parameters = parseConfig(config)
    problem_type = parameters.get("problem_type", None)
    problem_type = (
        find_problem_type_from_parameters(parameters)
        if problem_type is None
        else problem_type
    )
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
        for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
            current_subject_id = row[headers["subjectid"]]
            overall_stats_dict[current_subject_id] = {}
            label_image = torchio.LabelMap(row[headers["target"]])
            pred_image = torchio.LabelMap(row[headers["prediction"]])
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
                    current_prediction,
                    current_target,
                ).item()
                nsd, hd100, hd95 = _calculator_generic_all_surface_distances(
                    current_prediction,
                    current_target,
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
                    current_prediction,
                    current_target,
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
                    current_prediction,
                    current_target,
                    parameters,
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
            input_tensor,
            reference_tensor=None,
            p_min=0.5,
            p_max=99.5,
            strictlyPositive=True,
        ):
            """Normalizes a tensor based on percentiles. Clips values below and above the percentile.
            Percentiles for normalization can come from another tensor.

            Args:
                input_tensor (torch.Tensor): Tensor to be normalized based on the data from the reference_tensor.
                    If reference_tensor is None, the percentiles from this tensor will be used.
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
            v_min, v_max = np.percentile(
                reference_tensor, [p_min, p_max]
            )  # get p_min percentile and p_max percentile

            # set lower bound to be 0 if strictlyPositive is enabled
            v_min = max(v_min, 0.0) if strictlyPositive else v_min
            output_tensor = np.clip(
                input_tensor, v_min, v_max
            )  # clip values to percentiles from reference_tensor
            output_tensor = (output_tensor - v_min) / (
                v_max - v_min
            )  # normalizes values to [0;1]
            return output_tensor

        for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
            current_subject_id = row[headers["subjectid"]]
            overall_stats_dict[current_subject_id] = {}
            target_image = __fix_2d_tensor(
                torchio.ScalarImage(row[headers["target"]]).data
            )
            pred_image = __fix_2d_tensor(
                torchio.ScalarImage(row[headers["prediction"]]).data
            )
            # if "mask" is not in the row, we assume that the whole image is the mask
            # always cast to byte tensor
            mask = (
                __fix_2d_tensor(torchio.LabelMap(row[headers["mask"]]).data)
                if "mask" in row
                else torch.from_numpy(
                    np.ones(target_image.numpy().shape, dtype=np.uint8)
                )
            ).byte()

            # Get Infill region (we really are only interested in the infill region)
            output_infill = (pred_image * mask).float()
            gt_image_infill = (target_image * mask).float()

            # Normalize to [0;1] based on GT (otherwise MSE will depend on the image intensity range)
            normalize = parameters.get("normalize", True)
            if normalize:
                reference_tensor = (
                    target_image * ~mask
                )  # use all the tissue that is not masked for normalization
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
            ] = structural_similarity_index(gt_image_infill, output_infill, mask).item()

            # ncc metrics
            compute_ncc = parameters.get("compute_ncc", True)
            if compute_ncc:
                overall_stats_dict[current_subject_id]["ncc_mean"] = ncc_mean(
                    gt_image_infill, output_infill
                )
                overall_stats_dict[current_subject_id]["ncc_std"] = ncc_std(
                    gt_image_infill, output_infill
                )
                overall_stats_dict[current_subject_id]["ncc_max"] = ncc_max(
                    gt_image_infill, output_infill
                )
                overall_stats_dict[current_subject_id]["ncc_min"] = ncc_min(
                    gt_image_infill, output_infill
                )

            # only voxels that are to be inferred (-> flat array)
            # these are required for mse, psnr, etc.
            gt_image_infill = gt_image_infill[mask]
            output_infill = output_infill[mask]

            overall_stats_dict[current_subject_id]["mse"] = mean_squared_error(
                gt_image_infill, output_infill
            ).item()

            overall_stats_dict[current_subject_id]["msle"] = mean_squared_log_error(
                gt_image_infill, output_infill
            ).item()

            overall_stats_dict[current_subject_id]["mae"] = mean_absolute_error(
                gt_image_infill, output_infill
            ).item()

            # torchmetrics PSNR using "max"
            overall_stats_dict[current_subject_id]["psnr"] = peak_signal_noise_ratio(
                gt_image_infill, output_infill
            ).item()

            # same as above but with epsilon for robustness
            overall_stats_dict[current_subject_id][
                "psnr_eps"
            ] = peak_signal_noise_ratio(
                gt_image_infill, output_infill, epsilon=sys.float_info.epsilon
            ).item()

            # only use fix data range to [0;1] if the data was normalized before
            if normalize:
                # torchmetrics PSNR but with fixed data range of 0 to 1
                overall_stats_dict[current_subject_id][
                    "psnr_01"
                ] = peak_signal_noise_ratio(
                    gt_image_infill, output_infill, data_range=(0, 1)
                ).item()

                # same as above but with epsilon for robustness
                overall_stats_dict[current_subject_id][
                    "psnr_01_eps"
                ] = peak_signal_noise_ratio(
                    gt_image_infill,
                    output_infill,
                    data_range=(0, 1),
                    epsilon=sys.float_info.epsilon,
                ).item()

    pprint(overall_stats_dict)
    if outputfile is not None:
        with open(outputfile, "w") as outfile:
            yaml.dump(overall_stats_dict, outfile)
