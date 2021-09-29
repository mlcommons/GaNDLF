import os, pathlib, psutil
import torch
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import torchio

from GANDLF.utils import (
    get_date_time,
    get_filename_extension_sanitized,
    reverse_one_hot,
    resample_image,
)
from .step import step
from .loss_and_metric import get_loss_and_metrics


def validate_network(
    model, valid_dataloader, scheduler, params, epoch=0, mode="validation"
):
    """
    Function to validate a network for a single epoch

    Parameters
    ----------
    model : torch.model
        The model to process the input image with, it should support appropriate dimensions.
    valid_dataloader : torch.DataLoader
        The dataloader for the validation epoch
    params : dict
        The parameters passed by the user yaml
    mode: str
        The mode of validation, used to write outputs, if requested

    Returns
    -------
    average_epoch_valid_loss : float
        Validation loss for the current epoch
    average_epoch_valid_metric : dict
        Validation metrics for the current epoch

    """
    print("*" * 20)
    print("Starting " + mode + " : ")
    print("*" * 20)
    # Initialize a few things
    total_epoch_valid_loss = 0
    total_epoch_valid_metric = {}
    average_epoch_valid_metric = {}

    for metric in params["metrics"]:
        total_epoch_valid_metric[metric] = 0

    logits_list = []
    is_classification = params.get("problem_type") == "classification"
    is_inference = mode == "inference"

    # automatic mixed precision - https://pytorch.org/docs/stable/amp.html
    if params["verbose"]:
        if params["model"]["amp"]:
            print("Using Automatic mixed precision", flush=True)

    if scheduler is None:
        current_output_dir = params["output_dir"]  # this is in inference mode
    else:  # this is useful for inference
        current_output_dir = os.path.join(params["output_dir"], "output_" + mode)

    if not (is_inference):
        current_output_dir = os.path.join(current_output_dir, str(epoch))

    pathlib.Path(current_output_dir).mkdir(parents=True, exist_ok=True)

    # Set the model to valid
    model.eval()
    # # putting stuff in individual arrays for correlation analysis
    # all_targets = []
    # all_predics = []
    if params["medcam_enabled"]:
        model.enable_medcam()
        params["medcam_enabled"] = True

    if params["save_output"]:
        if "value_keys" in params:
            file_to_write = os.path.join(current_output_dir, "output_predictions.csv")
            if os.path.exists(file_to_write):
                # append to previously generated file
                file = open(file_to_write, "a")
                outputToWrite = ""
            else:
                # if file was absent, write header information
                file = open(file_to_write, "w")
                # used to write output
                outputToWrite = "Epoch,SubjectID,PredictedValue\n"

    if params["track_memory_usage"]:
        file_to_write_mem = os.path.join(current_output_dir, "memory_usage.csv")
        if os.path.exists(file_to_write_mem):
            # append to previously generated file
            file_mem = open(file_to_write_mem, "a")
            outputToWrite_mem = ""
        else:
            # if file was absent, write header information
            file_mem = open(file_to_write_mem, "w")
            outputToWrite_mem = "Epoch,Memory_Total,Memory_Available,Memory_Percent_Free,Memory_Usage,"  # used to write output
            if params["device"] == "cuda":
                outputToWrite_mem += "CUDA_active.all.current,CUDA_active.all.peak,"
            outputToWrite_mem += "\n"

        mem = psutil.virtual_memory()
        outputToWrite_mem += (
            str(epoch)
            + ","
            + str(mem[0])
            + ","
            + str(mem[1])
            + ","
            + str(mem[2])
            + ","
            + str(mem[3])
        )
        if params["device"] == "cuda":
            mem_cuda = torch.cuda.memory_stats()
            outputToWrite_mem += (
                ","
                + str(mem_cuda["active.all.current"])
                + ","
                + str(mem_cuda["active.all.peak"])
            )
        outputToWrite_mem += ",\n"
        file_mem.write(outputToWrite_mem)
        file_mem.close()

    for batch_idx, (subject) in enumerate(
        tqdm(valid_dataloader, desc="Looping over " + mode + " data")
    ):
        if params["verbose"]:
            print("== Current subject:", subject["subject_id"], flush=True)

        # ensure spacing is always present in params and is always subject-specific
        if "spacing" in subject:
            params["subject_spacing"] = subject["spacing"]
        else:
            params["subject_spacing"] = None

        # constructing a new dict because torchio.GridSampler requires torchio.Subject, which requires torchio.Image to be present in initial dict, which the loader does not provide
        subject_dict = {}
        label_ground_truth = None
        label_present = False
        # this is when we want the dataloader to pick up properties of GaNDLF's DataLoader, such as pre-processing and augmentations, if appropriate
        if "label" in subject:
            if subject["label"] != ["NA"]:
                subject_dict["label"] = torchio.Image(
                    path=subject["label"]["path"],
                    type=torchio.LABEL,
                    tensor=subject["label"]["data"].squeeze(0),
                )
                label_present = True
                label_ground_truth = subject_dict["label"]["data"]

        if "value_keys" in params:  # for regression/classification
            for key in params["value_keys"]:
                subject_dict["value_" + key] = subject[key]
                label_ground_truth = torch.cat(
                    [subject[key] for key in params["value_keys"]], dim=0
                )

        for key in params["channel_keys"]:
            subject_dict[key] = torchio.Image(
                path=subject[key]["path"],
                type=subject[key]["type"],
                tensor=subject[key]["data"].squeeze(0),
            )

        # regression/classification problem AND label is present
        if ("value_keys" in params) and label_present:
            sampler = torchio.data.LabelSampler(params["patch_size"])
            tio_subject = torchio.Subject(subject_dict)
            generator = sampler(tio_subject, num_patches=params["q_samples_per_volume"])
            pred_output = 0
            for patch in generator:
                image = torch.cat(
                    [patch[key][torchio.DATA] for key in params["channel_keys"]], dim=0
                )
                valuesToPredict = torch.cat(
                    [patch["value_" + key] for key in params["value_keys"]], dim=0
                )
                image = image.unsqueeze(0)
                image = image.float().to(params["device"])
                ## special case for 2D
                if image.shape[-1] == 1:
                    image = torch.squeeze(image, -1)
                pred_output += model(image)
            pred_output = pred_output.cpu() / params["q_samples_per_volume"]
            pred_output /= params["scaling_factor"]
            # all_predics.append(pred_output.double())
            # all_targets.append(valuesToPredict.double())
            print(f"pred_output.shape: {pred_output.shape}")

            if is_inference and is_classification:
                logits_list.append(pred_output)

            if params["save_output"]:
                outputToWrite += (
                    str(epoch)
                    + ","
                    + subject["subject_id"][0]
                    + ","
                    + str(pred_output.cpu().max().data.item())
                    + "\n"
                )
            final_loss, final_metric = get_loss_and_metrics(
                image, valuesToPredict, pred_output, params
            )
            # # Non network validing related
            total_epoch_valid_loss += final_loss.cpu().data.item()
            for metric in final_metric.keys():
                # calculated_metrics[metric]
                total_epoch_valid_metric[metric] += final_metric[metric]

        else:  # for segmentation problems OR regression/classification when no label is present
            grid_sampler = torchio.inference.GridSampler(
                torchio.Subject(subject_dict), params["patch_size"]
            )
            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
            aggregator = torchio.inference.GridAggregator(grid_sampler)

            if params["medcam_enabled"]:
                attention_map_aggregator = torchio.inference.GridAggregator(
                    grid_sampler
                )

            output_prediction = 0  # this is used for regression/classification
            current_patch = 0
            is_segmentation = True
            for patches_batch in patch_loader:
                if params["verbose"]:
                    print(
                        "=== Current patch:",
                        current_patch,
                        ", time : ",
                        get_date_time(),
                        ", location :",
                        patches_batch[torchio.LOCATION],
                        flush=True,
                    )
                current_patch += 1
                image = (
                    torch.cat(
                        [
                            patches_batch[key][torchio.DATA]
                            for key in params["channel_keys"]
                        ],
                        dim=1,
                    )
                    .float()
                    .to(params["device"])
                )
                if "value_keys" in params:
                    is_segmentation = False
                    label = label_ground_truth  # torch.cat([patches_batch[key] for key in params["value_keys"]], dim=0)
                    # label = torch.reshape(
                    #     patches_batch[params["value_keys"][0]], (params["batch_size"], 1)
                    # )
                    # one-hot encoding of 'label' will probably be needed for segmentation
                else:
                    label = patches_batch["label"][torchio.DATA]
                label = label.to(params["device"])
                if params["verbose"]:
                    print(
                        "=== Validation shapes : label:",
                        label.shape,
                        ", image:",
                        image.shape,
                        flush=True,
                    )

                result = step(model, image, label, params)

                # get the current attention map and add it to its aggregator
                if params["medcam_enabled"]:
                    _, _, output, attention_map = result
                    attention_map_aggregator.add_batch(
                        attention_map, patches_batch[torchio.LOCATION]
                    )
                else:
                    _, _, output = result

                if is_segmentation:
                    aggregator.add_batch(
                        output.detach().cpu(), patches_batch[torchio.LOCATION]
                    )
                else:
                    if torch.is_tensor(output):
                        # this probably needs customization for classification (majority voting or median, perhaps?)
                        output_prediction += output.detach().cpu()
                    else:
                        output_prediction += output

            # save outputs
            if is_segmentation:
                output_prediction = aggregator.get_output_tensor()
                output_prediction = output_prediction.unsqueeze(0)
                label_ground_truth = label_ground_truth.unsqueeze(0)
                if params["save_output"]:
                    path_to_metadata = subject["path_to_metadata"][0]
                    inputImage = sitk.ReadImage(path_to_metadata)
                    ext = get_filename_extension_sanitized(path_to_metadata)
                    pred_mask = output_prediction.numpy()
                    pred_mask = reverse_one_hot(
                        pred_mask[0], params["model"]["class_list"]
                    )
                    result_array = np.swapaxes(pred_mask, 0, 2)
                    ## special case for 2D
                    if image.shape[-1] > 1:
                        # ITK expects array as Z,X,Y
                        result_image = sitk.GetImageFromArray(result_array)
                    else:
                        result_image = sitk.GetImageFromArray(result_array.squeeze(0))
                    result_image.CopyInformation(inputImage)
                    # cast as the same data type
                    result_image = sitk.Cast(result_image, inputImage.GetPixelID())
                    # this handles cases that need resampling/resizing
                    if "resample" in params["data_preprocessing"]:
                        result_image = resample_image(
                            result_image,
                            inputImage.GetSpacing(),
                            interpolator=sitk.sitkNearestNeighbor,
                        )
                    sitk.WriteImage(
                        result_image,
                        os.path.join(
                            current_output_dir, subject["subject_id"][0] + "_seg" + ext
                        ),
                    )
                # reverse one-hot encoding of 'output_prediction' will probably be needed for segmentation
            else:
                output_prediction = output_prediction / len(
                    patch_loader
                )  # final regression output
                if params["save_output"]:
                    outputToWrite += (
                        str(epoch)
                        + ","
                        + subject["subject_id"][0]
                        + ","
                        + str(output_prediction)
                        + "\n"
                    )

            # get the final attention map and save it
            if params["medcam_enabled"]:
                attention_map = attention_map_aggregator.get_output_tensor()
                for i, n in enumerate(attention_map):
                    model.save_attention_map(
                        n.squeeze(), raw_input=image[i].squeeze(-1)
                    )

            output_prediction = output_prediction.squeeze(-1)
            if is_inference and is_classification:
                logits_list.append(output_prediction)

            final_loss, final_metric = get_loss_and_metrics(
                image, label_ground_truth, output_prediction, params
            )
            if params["verbose"]:
                print(
                    "Full image " + mode + ":: Loss: ",
                    final_loss,
                    "; Metric: ",
                    final_metric,
                    flush=True,
                )

            # # Non network validing related
            # loss.cpu().data.item()
            total_epoch_valid_loss += final_loss.cpu().data.item()
            for metric in final_metric.keys():
                # calculated_metrics[metric]
                total_epoch_valid_metric[metric] += final_metric[metric]

        # For printing information at halftime during an epoch
        if ((batch_idx + 1) % (len(valid_dataloader) / 2) == 0) and (
            (batch_idx + 1) < len(valid_dataloader)
        ):
            print(
                "\nHalf-Epoch Average " + mode + " loss : ",
                total_epoch_valid_loss / (batch_idx + 1),
            )
            for metric in params["metrics"]:
                print(
                    "Half-Epoch Average " + mode + " " + metric + " : ",
                    total_epoch_valid_metric[metric] / (batch_idx + 1),
                )

    if params["medcam_enabled"]:
        model.disable_medcam()
        params["medcam_enabled"] = False

    average_epoch_valid_loss = total_epoch_valid_loss / len(valid_dataloader)
    print("     Epoch Final   " + mode + " loss : ", average_epoch_valid_loss)
    for metric in params["metrics"]:
        average_epoch_valid_metric[metric] = total_epoch_valid_metric[metric] / len(
            valid_dataloader
        )
        print(
            "     Epoch Final   " + mode + " " + metric + " : ",
            average_epoch_valid_metric[metric],
        )

    if scheduler is not None:
        if params["scheduler"]["type"] in [
            "reduce_on_plateau",
            "reduce-on-plateau",
            "plateau",
            "reduceonplateau",
        ]:
            scheduler.step(average_epoch_valid_loss)
        else:
            scheduler.step()

    # write the predictions, if appropriate
    if params["save_output"]:
        if is_inference and is_classification and logits_list:
            logit_tensor = torch.cat(logits_list)
            current_fold_dir = params["current_fold_dir"]
            np.savetxt(
                os.path.join(current_fold_dir, "logits.csv"),
                logit_tensor.detach().numpy(),
                delimiter=",",
            )

        if "value_keys" in params:
            file.write(outputToWrite)
            file.close()

    return average_epoch_valid_loss, average_epoch_valid_metric
