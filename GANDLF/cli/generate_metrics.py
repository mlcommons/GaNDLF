import yaml, math, sys
from pprint import pprint
import pandas as pd
from tqdm import tqdm
import torch
import torchio
import SimpleITK as sitk
import numpy as np
from torchmetrics import (
    StructuralSimilarityIndexMeasure,
    MeanSquaredError,
)

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
        for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
            current_subject_id = row[headers["subjectid"]]
            overall_stats_dict[current_subject_id] = {}
            target_image = sitk.ReadImage(row["target"])
            pred_image = sitk.ReadImage(row["prediction"])
            # special case for vector (i.e, RGB) images - there is probably a better way to do this
            if "vector" in target_image.GetPixelIDTypeAsString().lower():
                target_image = sitk.ReadImage(row["target"], sitk.sitkFloat32)
            if "vector" in pred_image.GetPixelIDTypeAsString().lower():
                pred_image = sitk.ReadImage(row["prediction"], sitk.sitkFloat32)
            stats_filter = sitk.StatisticsImageFilter()
            stats_filter.Execute(target_image)
            max_target = stats_filter.GetMaximum()
            min_target = stats_filter.GetMinimum()
            sq_diff = sitk.SquaredDifference(target_image, pred_image)
            stats_filter.Execute(sq_diff)
            overall_stats_dict[current_subject_id]["mse"] = stats_filter.GetMean()
            overall_stats_dict[current_subject_id]["rmse"] = stats_filter.GetSigma()

            # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
            # subtracting min_target from max_target to get the dynamic range
            # added epsilon to avoid divide by zero
            overall_stats_dict[current_subject_id]["psnr"] = 10 * math.log10(
                (max_target - min_target) ** 2
                / (
                    overall_stats_dict[current_subject_id]["mse"]
                    + sys.float_info.epsilon
                )
            )

            sii_filter = sitk.SimilarityIndexImageFilter()
            sii_filter.Execute(target_image, pred_image)
            overall_stats_dict[current_subject_id][
                "ssim"
            ] = sii_filter.GetSimilarityIndex()

            correlation_filter = sitk.FFTNormalizedCorrelationImageFilter()
            corr_image = correlation_filter.Execute(target_image, pred_image)
            stats_filter.Execute(corr_image)
            overall_stats_dict[current_subject_id][
                "ncc_fft_mean"
            ] = stats_filter.GetMean()
            overall_stats_dict[current_subject_id][
                "ncc_fft_std"
            ] = stats_filter.GetSigma()
            overall_stats_dict[current_subject_id][
                "ncc_fft_max"
            ] = stats_filter.GetMaximum()
            overall_stats_dict[current_subject_id][
                "ncc_fft_min"
            ] = stats_filter.GetMinimum()

    elif problem_type == "brats_synthesis":

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

        for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
            current_subject_id = row[headers["subjectid"]]
            overall_stats_dict[current_subject_id] = {}
            target_image = __fix_2d_tensor(torchio.ScalarImage(row["target"]).data)
            pred_image = __fix_2d_tensor(torchio.ScalarImage(row["prediction"]).data)
            mask = (
                __fix_2d_tensor(torchio.LabelMap(row["mask"]).data)
                if "mask" in row
                else torch.from_numpy(
                    np.ones(target_image.numpy().shape, dtype=np.uint8)
                )
            )

            # Define evaluation Metrics
            # psnr = PeakSignalNoiseRatio()
            ssim = StructuralSimilarityIndexMeasure(return_full_image=True)
            mse = MeanSquaredError()

            # Get Infill region (we really are only interested in the infill region)
            output_infill = (pred_image * mask).float()
            gt_image_infill = (target_image * mask).float()

            # Normalize to [0;1] based on GT (otherwise MSE will depend on the image intensity range)
            normalize = parameters.get("normalize", True)
            if normalize:
                v_max = gt_image_infill.max()
                output_infill /= v_max
                gt_image_infill /= v_max

            # SSIM - apply on complete masked image but only take values from masked region
            _, ssim_idx_full_image = ssim(gt_image_infill, output_infill)
            try:
                ssim_idx = ssim_idx_full_image[mask]
            except Exception as e:
                print(f"Error: {e}")
                if len(ssim_idx_full_image.shape) == 0:
                    ssim_idx = torch.ones_like(mask) * ssim_idx_full_image
            overall_stats_dict[current_subject_id]["ssim"] = ssim_idx.mean().item()

            # only voxels that are to be inferred (-> flat array)
            gt_image_infill = gt_image_infill[mask]
            output_infill = output_infill[mask]

            # MSE
            overall_stats_dict[current_subject_id]["mse"] = mse(
                gt_image_infill, output_infill
            ).item()

            # PSNR - similar to pytorch PeakSignalNoiseRatio until 4 digits after decimal point
            overall_stats_dict[current_subject_id]["psnr"] = (
                10.0
                * torch.log10(
                    (torch.max(gt_image_infill) - torch.min(gt_image_infill)) ** 2
                    / overall_stats_dict[current_subject_id]["mse"]
                ).item()
            )

    pprint(overall_stats_dict)
    if outputfile is not None:
        with open(outputfile, "w") as outfile:
            yaml.dump(overall_stats_dict, outfile)
