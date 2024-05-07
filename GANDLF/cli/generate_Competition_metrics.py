import numpy as np
import nibabel as nib
import cc3d
import scipy
import pandas as pd
from GANDLF.cli import surface_distance
import sys
import math
import yaml
import warnings
from pprint import pprint

warnings.simplefilter(action="ignore", category=FutureWarning)


def dice(im1, im2):
    """
    Computes Dice score for two images

    Parameters
    ==========
    im1: Numpy Array/Matrix; Predicted segmentation in matrix form
    im2: Numpy Array/Matrix; Ground truth segmentation in matrix form

    Output
    ======
    dice_score: Dice score between two images
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError(
            "Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * (intersection.sum()) / (im1.sum() + im2.sum())


def get_TissueWiseSeg(prediction_matrix, gt_matrix, tissue_type):
    """
    Converts the segmentatations to isolate tissue types

    Parameters
    ==========
    prediction_matrix: Numpy Array/Matrix; Predicted segmentation in matrix form
    gt_matrix: Numpy Array/Matrix; Ground truth segmentation in matrix form
    tissue_type: str; Can be WT, ET or TC

    Output
    ======
    prediction_matrix: Numpy Array/Matrix; Predicted segmentation in matrix form with
                       just tissue type mentioned
    gt_matrix: Numpy Array/Matrix; Ground truth segmentation in matrix form with just
                       tissue type mentioned
    """

    if tissue_type == "WT":
        np.place(
            prediction_matrix,
            (prediction_matrix != 1)
            & (prediction_matrix != 2)
            & (prediction_matrix != 3),
            0,
        )
        np.place(prediction_matrix, (prediction_matrix > 0), 1)

        np.place(gt_matrix, (gt_matrix != 1) & (
            gt_matrix != 2) & (gt_matrix != 3), 0)
        np.place(gt_matrix, (gt_matrix > 0), 1)

    elif tissue_type == "TC":
        np.place(
            prediction_matrix, (prediction_matrix != 1) & (
                prediction_matrix != 3), 0
        )
        np.place(prediction_matrix, (prediction_matrix > 0), 1)

        np.place(gt_matrix, (gt_matrix != 1) & (gt_matrix != 3), 0)
        np.place(gt_matrix, (gt_matrix > 0), 1)

    elif tissue_type == "ET":
        np.place(prediction_matrix, (prediction_matrix != 3), 0)
        np.place(prediction_matrix, (prediction_matrix > 0), 1)

        np.place(gt_matrix, (gt_matrix != 3), 0)
        np.place(gt_matrix, (gt_matrix > 0), 1)

    return prediction_matrix, gt_matrix


def get_GTseg_combinedByDilation(gt_dilated_cc_mat, gt_label_cc):
    """
    Computes the Corrected Connected Components after combing lesions
    together with respect to their dilation extent

    Parameters
    ==========
    gt_dilated_cc_mat: Numpy Array/Matrix; Ground Truth Dilated Segmentation
                       after CC Analysis
    gt_label_cc: Numpy Array/Matrix; Ground Truth Segmentation after
                       CC Analysis

    Output
    ======
    gt_seg_combinedByDilation_mat: Numpy Array/Matrix; Ground Truth
                                   Segmentation after CC Analysis and
                                   combining lesions
    """

    gt_seg_combinedByDilation_mat = np.zeros_like(gt_dilated_cc_mat)

    for comp in range(np.max(gt_dilated_cc_mat)):
        comp += 1

        gt_d_tmp = np.zeros_like(gt_dilated_cc_mat)
        gt_d_tmp[gt_dilated_cc_mat == comp] = 1
        gt_d_tmp = gt_label_cc * gt_d_tmp

        np.place(gt_d_tmp, gt_d_tmp > 0, comp)
        gt_seg_combinedByDilation_mat += gt_d_tmp

    return gt_seg_combinedByDilation_mat


def get_LesionWiseScores(prediction_seg, gt_seg, label_value, dil_factor):
    """
    Computes the Lesion-wise scores for pair of prediction and ground truth
    segmentations

    Parameters
    ==========
    prediction_seg: str; location of the prediction segmentation
    gt_label_cc: str; location of the gt segmentation
    label_value: str; Can be WT, ET or TC
    dil_factor: int; Used to perform dilation

    Output
    ======
    tp: Number of TP lesions WRT prediction segmentation
    fn: Number of FN lesions WRT prediction segmentation
    fp: Number of FP lesions WRT prediction segmentation
    gt_tp: Number of Ground Truth TP lesions WRT prediction segmentation
    metric_pairs: list; All the lesion-wise metrics
    full_dice: Dice Score of the pair of segmentations
    full_gt_vol: Total Ground Truth Segmenatation Volume
    full_pred_vol: Total Prediction Segmentation Volume
    """

    # Get Prediction and GT segs matrix files
    pred_nii = nib.load(prediction_seg)
    gt_nii = nib.load(gt_seg)
    pred_mat = pred_nii.get_fdata()
    gt_mat = gt_nii.get_fdata()

    # Get Spacing to computes volumes
    # Brats Assumes all spacing is 1x1x1mm3
    sx, sy, sz = pred_nii.header.get_zooms()

    # Get the prediction and GT matrix based on
    # WT, TC, ET

    pred_mat, gt_mat = get_TissueWiseSeg(
        prediction_matrix=pred_mat, gt_matrix=gt_mat, tissue_type=label_value
    )

    # Get Dice score for the full image
    if np.all(gt_mat == 0) and np.all(pred_mat == 0):
        full_dice = 1.0
    else:
        full_dice = dice(pred_mat, gt_mat)

    # Get HD95 sccre for the full image

    if np.all(gt_mat == 0) and np.all(pred_mat == 0):
        full_hd95 = 0.0
    else:
        full_sd = surface_distance.compute_surface_distances(
            gt_mat.astype(int), pred_mat.astype(int), (sx, sy, sz)
        )
        full_hd95 = surface_distance.compute_robust_hausdorff(full_sd, 95)

    # Get Sensitivity and Specificity
    full_sens, full_specs = get_sensitivity_and_specificity(
        result_array=pred_mat, target_array=gt_mat
    )

    # Get GT Volume and Pred Volume for the full image
    full_gt_vol = np.sum(gt_mat) * sx * sy * sz
    full_pred_vol = np.sum(pred_mat) * sx * sy * sz

    # Performing Dilation and CC analysis

    dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)

    gt_mat_cc = cc3d.connected_components(gt_mat, connectivity=26)
    pred_mat_cc = cc3d.connected_components(pred_mat, connectivity=26)

    gt_mat_dilation = scipy.ndimage.binary_dilation(
        gt_mat, structure=dilation_struct, iterations=dil_factor
    )
    gt_mat_dilation_cc = cc3d.connected_components(
        gt_mat_dilation, connectivity=26)

    gt_mat_combinedByDilation = get_GTseg_combinedByDilation(
        gt_dilated_cc_mat=gt_mat_dilation_cc, gt_label_cc=gt_mat_cc
    )

    # Performing the Lesion-By-Lesion Comparison

    gt_label_cc = gt_mat_combinedByDilation
    pred_label_cc = pred_mat_cc

    gt_tp = []
    tp = []
    fn = []
    fp = []
    metric_pairs = []

    for gtcomp in range(np.max(gt_label_cc)):
        gtcomp += 1

        # Extracting current lesion
        gt_tmp = np.zeros_like(gt_label_cc)
        gt_tmp[gt_label_cc == gtcomp] = 1

        # Extracting ROI GT lesion component
        gt_tmp_dilation = scipy.ndimage.binary_dilation(
            gt_tmp, structure=dilation_struct, iterations=dil_factor
        )

        # Volume of lesion
        gt_vol = np.sum(gt_tmp) * sx * sy * sz

        # Extracting Predicted true positive lesions
        pred_tmp = np.copy(pred_label_cc)
        # pred_tmp = pred_tmp*gt_tmp
        pred_tmp = pred_tmp * gt_tmp_dilation
        intersecting_cc = np.unique(pred_tmp)
        intersecting_cc = intersecting_cc[intersecting_cc != 0]
        for cc in intersecting_cc:
            tp.append(cc)

        # Isolating Predited Lesions to calulcate Metrics
        pred_tmp = np.copy(pred_label_cc)
        pred_tmp[np.isin(pred_tmp, intersecting_cc, invert=True)] = 0
        pred_tmp[np.isin(pred_tmp, intersecting_cc)] = 1

        # Calculating Lesion-wise Dice and HD95
        dice_score = dice(pred_tmp, gt_tmp)
        surface_distances = surface_distance.compute_surface_distances(
            gt_tmp, pred_tmp, (sx, sy, sz)
        )
        hd = surface_distance.compute_robust_hausdorff(surface_distances, 95)

        metric_pairs.append((intersecting_cc, gtcomp, gt_vol, dice_score, hd))

        # Extracting Number of TP/FP/FN and other data
        if len(intersecting_cc) > 0:
            gt_tp.append(gtcomp)
        else:
            fn.append(gtcomp)

    fp = np.unique(pred_label_cc[np.isin(
        pred_label_cc, tp + [0], invert=True)])

    return (
        tp,
        fn,
        fp,
        gt_tp,
        metric_pairs,
        full_dice,
        full_hd95,
        full_gt_vol,
        full_pred_vol,
        full_sens,
        full_specs,
    )


def get_sensitivity_and_specificity(result_array, target_array):
    """
    This function is extracted from GaNDLF from mlcommons

    You can find the documentation here -

    https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/metrics/segmentation.py#L196

    """
    iC = np.sum(result_array)
    rC = np.sum(target_array)

    overlap = np.where((result_array == target_array), 1, 0)

    # Where they agree are both equal to that value
    TP = overlap[result_array == 1].sum()
    FP = iC - TP
    FN = rC - TP
    TN = np.count_nonzero((result_array != 1) & (target_array != 1))

    Sens = 1.0 * TP / (TP + FN + sys.float_info.min)
    Spec = 1.0 * TN / (TN + FP + sys.float_info.min)

    # Make Changes if both input and reference are 0 for the tissue type
    if (iC == 0) and (rC == 0):
        Sens = 1.0

    return Sens, Spec


def get_LesionWiseResults(pred_file, gt_file, challenge_name, output=None):
    """
    Computes the Lesion-wise scores for pair of prediction and ground truth
    segmentations

    Parameters
    ==========
    pred_file: str; location of the prediction segmentation
    gt_file: str; location of the gt segmentation
    challenge_name: str; name of the challenge for parameters


    Output
    ======
    Saves the performance metrics as CSVs
    results_df: pd.DataFrame; lesion-wise results with other metrics
    """

    # Dilation and Threshold Parameters
    if challenge_name == "BraTS-GLI":
        dilation_factor = 3
        lesion_volume_thresh = 50
    elif challenge_name == "BraTS-SSA":
        dilation_factor = 3
        lesion_volume_thresh = 50
    elif challenge_name == "BraTS-MEN":
        dilation_factor = 1
        lesion_volume_thresh = 50
    elif challenge_name == "BraTS-PED":
        dilation_factor = 3
        lesion_volume_thresh = 50
    elif challenge_name == "BraTS-MET":
        dilation_factor = 1
        lesion_volume_thresh = 2
    elif challenge_name == "FeTS-2024":
        dilation_factor = 3
        lesion_volume_thresh = 50

    final_lesionwise_metrics_df = pd.DataFrame()
    final_metrics_dict = dict()
    label_values = ["WT", "TC", "ET"]

    for l in range(len(label_values)):
        (
            _,
            fn,
            fp,
            gt_tp,
            metric_pairs,
            full_dice,
            full_hd95,
            full_gt_vol,
            _,
            full_sens,
            full_specs,
        ) = get_LesionWiseScores(
            prediction_seg=pred_file,
            gt_seg=gt_file,
            label_value=label_values[l],
            dil_factor=dilation_factor,
        )

        metric_df = (
            pd.DataFrame(
                metric_pairs,
                columns=[
                    "predicted_lesion_numbers",
                    "gt_lesion_numbers",
                    "gt_lesion_vol",
                    "dice_lesionwise",
                    "hd95_lesionwise",
                ],
            )
            .sort_values(by=["gt_lesion_numbers"], ascending=True)
            .reset_index(drop=True)
        )

        metric_df["_len"] = metric_df["predicted_lesion_numbers"].map(len)

        # Removing <= Threshold volume lesions from analysis
        fn_sub = (
            metric_df[
                (metric_df["_len"] == 0)
                & (metric_df["gt_lesion_vol"] <= lesion_volume_thresh)
            ]
        ).shape[0]

        gt_tp_sub = (
            metric_df[
                (metric_df["_len"] != 0)
                & (metric_df["gt_lesion_vol"] <= lesion_volume_thresh)
            ]
        ).shape[0]

        metric_df["Label"] = [label_values[l]] * len(metric_df)
        metric_df["hd95_lesionwise"] = metric_df["hd95_lesionwise"].replace(
            np.inf, 374)

        # final_lesionwise_metrics_df = final_lesionwise_metrics_df.append(
        #   metric_df)

        final_lesionwise_metrics_df = pd.concat(
            [final_lesionwise_metrics_df, metric_df], ignore_index=True
        )

        metric_df_thresh = metric_df[metric_df["gt_lesion_vol"]
                                     > lesion_volume_thresh]

        try:
            lesion_wise_dice = np.sum(metric_df_thresh["dice_lesionwise"]) / (
                len(metric_df_thresh) + len(fp)
            )
        except ZeroDivisionError:
            lesion_wise_dice = np.nan

        try:
            lesion_wise_hd95 = (
                np.sum(metric_df_thresh["hd95_lesionwise"]) + len(fp) * 374
            ) / (len(metric_df_thresh) + len(fp))
        except:
            lesion_wise_hd95 = np.nan

        if math.isnan(lesion_wise_dice):
            lesion_wise_dice = 1

        if math.isnan(lesion_wise_hd95):
            lesion_wise_hd95 = 0

        metrics_dict = {
            "Num_TP": len(gt_tp) - gt_tp_sub,  # GT_TP
            # 'Num_TP' : len(tp),
            "Num_FP": len(fp),
            "Num_FN": len(fn) - fn_sub,
            "Sensitivity": full_sens,
            "Specificity": full_specs,
            "Legacy_Dice": full_dice,
            "Legacy_HD95": full_hd95,
            "GT_Complete_Volume": full_gt_vol,
            "LesionWise_Score_Dice": lesion_wise_dice,
            "LesionWise_Score_HD95": lesion_wise_hd95,
        }

        final_metrics_dict[label_values[l]] = metrics_dict

    # final_lesionwise_metrics_df.to_csv(os.path.split(pred_file)[0] + '/' +
    #                                   os.path.split(pred_file)[1].split('.')[0] +
    #                                   '_lesionwise_metrics.csv',
    #                                   index=False)

    results_df = pd.DataFrame(final_metrics_dict).T
    results_df["Labels"] = results_df.index
    results_df = results_df.reset_index(drop=True)
    results_df.insert(0, "Labels", results_df.pop("Labels"))
    results_df.replace(np.inf, 374, inplace=True)

    if output:
        results_df.to_csv(output, index=False)

    results_df.index = results_df["Labels"]
    results_df = results_df.drop(["Labels"], axis=1)

    results_dict = results_df.T.to_dict()

    return results_dict


def generate_metrics_dict_competition(input_data, challenge, output_file=None):
    input_df = pd.read_csv(input_data)

    overall_stats_dict = {}
    for i in range(len(input_df)):
        lesionwise_dict = get_LesionWiseResults(
            pred_file=input_df["prediction"][i],
            gt_file=input_df["target"][i],
            challenge_name=challenge,
            output=None,
        )

        overall_stats_dict[input_df["subjectid"][i]] = lesionwise_dict

    pprint(overall_stats_dict)
    if output_file is not None:
        with open(output_file, "w") as outfile:
            yaml.dump(overall_stats_dict, outfile)

    return overall_stats_dict
