#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 21:45:06 2021

@author: siddhesh
"""
import os, math
import torch
import time
import torchio
import psutil
from torch.utils.data import DataLoader
from GANDLF.logger import Logger
from GANDLF.losses import one_hot
from GANDLF.parameterParsing import (
    get_model,
    get_optimizer,
    get_scheduler,
    get_loss_and_metrics,
)
from GANDLF.utils import (
    get_date_time,
    resample_image,
    send_model_to_device,
    populate_channel_keys_in_params,
    reverse_one_hot,
    get_class_imbalance_weights,
)
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
import SimpleITK as sitk
import numpy as np
from medcam import medcam

os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = "1"  # hides torchio citation request

# Reminder, the scaling factor should go to the metric MSE, and all should support a scaling factor, right?


def step(model, image, label, params):
    """
    Function that steps the model for a single batch

    Parameters
    ----------
    model : torch.model
        The model to process the input image with, it should support appropriate dimensions.
    image : torch.Tensor
        The input image stack according to requirements
    label : torch.Tensor
        The input label for the corresponding image label
    params : dict
        The parameters passed by the user yaml

    Returns
    -------
    loss : torch.Tensor
        The computed loss from the label and the output
    metric_output : torch.Tensor
        The computed metric from the label and the output
    output: torch.Tensor
        The final output of the model

    """
    if params["verbose"]:
        # print('=== Memory (allocated; cached) : ', round(torch.cuda.memory_allocated()/1024**3, 1), '; ', round(torch.cuda.memory_reserved()/1024**3, 1))
        print(torch.cuda.memory_summary())
        print(
            "|===========================================================================|\n|                             CPU Utilization                            |\n|"
        )
        print("Load_Percent   :", psutil.cpu_percent(interval=None))
        print("MemUtil_Percent:", psutil.virtual_memory()[2])
        print(
            "|===========================================================================|\n|"
        )

    if params["model"]["dimension"] == 2:
        image = torch.squeeze(image, -1)
        if "value_keys" in params:  # squeeze label for segmentation only
            label = torch.squeeze(label, -1)
    if params["model"]["amp"]:
        with torch.cuda.amp.autocast():
            output = model(image)
    else:
        output = model(image)

    if "medcam_enabled" in params and params["medcam_enabled"]:
        output, attention_map = output

    # print("Output shape : ", output.shape, flush=True)
    # one-hot encoding of 'output' will probably be needed for segmentation
    loss, metric_output = get_loss_and_metrics(label, output, params)

    if params["model"]["dimension"] == 2:
        output = torch.unsqueeze(output, -1)
        if "medcam_enabled" in params and params["medcam_enabled"]:
            attention_map = torch.unsqueeze(attention_map, -1)

    if not ("medcam_enabled" in params and params["medcam_enabled"]):
        return loss, metric_output, output
    else:
        return loss, metric_output, output, attention_map


def train_network(model, train_dataloader, optimizer, params):
    """
    Function to train a network for a single epoch

    Parameters
    ----------
    model : torch.model
        The model to process the input image with, it should support appropriate dimensions.
    train_dataloader : torch.DataLoader
        The dataloader for the training epoch
    optimizer : torch.optim
        Optimizer for optimizing network
    params : dict
        the parameters passed by the user yaml

    Returns
    -------
    average_epoch_train_loss : float
        Train loss for the current epoch
    average_epoch_train_metric : dict
        Train metrics for the current epoch

    """
    # Initialize a few things
    total_epoch_train_loss = 0
    total_epoch_train_metric = {}
    average_epoch_train_metric = {}

    for metric in params["metrics"]:
        total_epoch_train_metric[metric] = 0

    # automatic mixed precision - https://pytorch.org/docs/stable/amp.html
    if params["model"]["amp"]:
        print("Using Automatic mixed precision", flush=True)
        scaler = torch.cuda.amp.GradScaler()

    # Fetch the optimizer

    # Set the model to train
    model.train()
    for batch_idx, (subject) in enumerate(train_dataloader):
        optimizer.zero_grad()
        image = (
            torch.cat(
                [subject[key][torchio.DATA] for key in params["channel_keys"]], dim=1
            )
            .float()
            .to(params["device"])
        )
        if "value_keys" in params:
            label = torch.cat([subject[key] for key in params["value_keys"]], dim=0)
            label = label.reshape(params["batch_size"], len(params["value_keys"]))
        else:
            label = subject["label"][torchio.DATA]
            # one-hot encoding of 'label' will probably be needed for segmentation
        label = label.to(params["device"])
        # print("Train : ", label.shape, image.shape, flush=True)
        loss, calculated_metrics, _ = step(model, image, label, params)
        nan_loss = True
        if params["model"]["amp"]:
            with torch.cuda.amp.autocast():
                if not torch.isnan(
                    loss
                ):  # if loss is nan, don't backprop and don't step optimizer
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    nan_loss = False
        else:
            if not math.isnan(loss):
                loss.backward()
                optimizer.step()
                nan_loss = False

        # Non network training related
        if not nan_loss:
            total_epoch_train_loss += loss.cpu().data.item()
        for metric in calculated_metrics.keys():
            total_epoch_train_metric[metric] += calculated_metrics[metric]

        # For printing information at halftime during an epoch
        if ((batch_idx + 1) % (len(train_dataloader) / 2) == 0) and (
            (batch_idx + 1) < len(train_dataloader)
        ):
            print(
                "Half-Epoch Average Train loss : ",
                total_epoch_train_loss / (batch_idx + 1),
            )
            for metric in params["metrics"]:
                print(
                    "Half-Epoch Average Train " + metric + " : ",
                    total_epoch_train_metric[metric] / (batch_idx + 1),
                )

    average_epoch_train_loss = total_epoch_train_loss / len(train_dataloader)
    print("     Epoch Final   Train loss : ", average_epoch_train_loss)
    for metric in params["metrics"]:
        average_epoch_train_metric[metric] = total_epoch_train_metric[metric] / len(
            train_dataloader
        )
        print(
            "     Epoch Final   Train " + metric + " : ",
            average_epoch_train_metric[metric],
        )

    return average_epoch_train_loss, average_epoch_train_metric


def validate_network(model, valid_dataloader, scheduler, params, mode="validation"):
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

    # automatic mixed precision - https://pytorch.org/docs/stable/amp.html
    if params["verbose"]:
        if params["model"]["amp"]:
            print("Using Automatic mixed precision", flush=True)

    if scheduler is None:
        current_output_dir = params["output_dir"]  # this is in inference mode
    else:  # this is useful for inference
        current_output_dir = os.path.join(params["output_dir"], mode + "_output")

    # Set the model to valid
    model.eval()
    # # putting stuff in individual arrays for correlation analysis
    # all_targets = []
    # all_predics = []
    if params["medcam_enabled"]:
        model.enable_medcam()
        params["medcam_enabled"] = True

    for batch_idx, (subject) in enumerate(valid_dataloader):
        if params["verbose"]:
            print("== Current subject:", subject["subject_id"], flush=True)

        # constructing a new dict because torchio.GridSampler requires torchio.Subject, which requires torchio.Image to be present in initial dict, which the loader does not provide
        subject_dict = {}
        label_ground_truth = None
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
                outputToWrite = "SubjectID,PredictedValue\n"  # used to write output

        for key in params["channel_keys"]:
            subject_dict[key] = torchio.Image(
                path=subject[key]["path"],
                type=subject[key]["type"],
                tensor=subject[key]["data"].squeeze(0),
            )

        if (
            "value_keys" in params
        ) and label_present:  # regression/classification problem AND label is present
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
            outputToWrite += (
                subject["subject_id"][0]
                + ","
                + str(pred_output.cpu().data.item())
                + "\n"
            )
            final_loss, final_metric = get_loss_and_metrics(
                valuesToPredict, pred_output, params
            )
            # # Non network validing related
            total_epoch_valid_loss += (
                final_loss.cpu().data.item()
            )  # loss.cpu().data.item()
            for metric in final_metric.keys():
                total_epoch_valid_metric[metric] += final_metric[
                    metric
                ]  # calculated_metrics[metric]

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
                        output_prediction += (
                            output.detach().cpu()
                        )  # this probably needs customization for classification (majority voting or median, perhaps?)
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
                    _, ext = os.path.splitext(path_to_metadata)
                    if (ext == ".gz") or (ext == ".nii"):
                        ext = ".nii.gz"
                    pred_mask = output_prediction.numpy()
                    pred_mask = reverse_one_hot(
                        pred_mask[0], params["model"]["class_list"]
                    )
                    ## special case for 2D
                    if image.shape[-1] > 1:
                        result_image = sitk.GetImageFromArray(
                            np.swapaxes(pred_mask, 0, 2)
                        )  # ITK expects array as Z,X,Y
                    else:
                        result_image = pred_mask
                    result_image.CopyInformation(inputImage)
                    result_image = sitk.Cast(
                        result_image, inputImage.GetPixelID()
                    )  # cast as the same data type
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
                outputToWrite += (
                    subject["subject_id"][0] + "," + str(output_prediction) + "\n"
                )

            # get the final attention map and save it
            if params["medcam_enabled"]:
                attention_map = attention_map_aggregator.get_output_tensor()
                for i in range(len(attention_map)):
                    model.save_attention_map(
                        attention_map[i].squeeze(), raw_input=image[i].squeeze(-1)
                    )

            final_loss, final_metric = get_loss_and_metrics(
                label_ground_truth, output_prediction, params
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
            total_epoch_valid_loss += (
                final_loss.cpu().data.item()
            )  # loss.cpu().data.item()
            for metric in final_metric.keys():
                total_epoch_valid_metric[metric] += final_metric[
                    metric
                ]  # calculated_metrics[metric]

        # For printing information at halftime during an epoch
        if ((batch_idx + 1) % (len(valid_dataloader) / 2) == 0) and (
            (batch_idx + 1) < len(valid_dataloader)
        ):
            print(
                "Half-Epoch Average " + mode + " loss : ",
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
        if params["scheduler"] == "reduce-on-plateau":
            scheduler.step(average_epoch_valid_loss)
        else:
            scheduler.step()

    # write the predictions, if appropriate
    if params["save_output"]:
        if "value_keys" in params:
            file = open(os.path.join(current_output_dir, "output_predictions.csv"), "w")
            file.write(outputToWrite)
            file.close()

    return average_epoch_valid_loss, average_epoch_valid_metric


def training_loop(
    training_data,
    validation_data,
    device,
    params,
    output_dir,
    testing_data=None,
):

    # Some autodetermined factors
    epochs = params["num_epochs"]
    loss = params["loss_function"]
    metrics = params["metrics"]
    params["device"] = device
    params["output_dir"] = output_dir

    # Defining our model here according to parameters mentioned in the configuration file
    print("Number of channels : ", params["model"]["num_channels"])

    # Fetch the model according to params mentioned in the configuration file
    model, params["model"]["amp"] = get_model(
        modelname=params["model"]["architecture"],
        num_dimensions=params["model"]["dimension"],
        num_channels=params["model"]["num_channels"],
        num_classes=params["model"]["num_classes"],
        base_filters=params["model"]["base_filters"],
        final_convolution_layer=params["model"]["final_layer"],
        patch_size=params["patch_size"],
        batch_size=params["batch_size"],
        amp=params["model"]["amp"],
    )

    # Set up the dataloaders
    training_data_for_torch = ImagesFromDataFrame(training_data, params, train=True)

    validation_data_for_torch = ImagesFromDataFrame(
        validation_data, params, train=False
    )

    testingDataDefined = True
    if testing_data is None:
        # testing_data = validation_data
        testingDataDefined = False

    if testingDataDefined:
        test_data_for_torch = ImagesFromDataFrame(testing_data, params, train=False)

    train_dataloader = DataLoader(
        training_data_for_torch,
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )

    val_dataloader = DataLoader(
        validation_data_for_torch,
        batch_size=1,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )

    if testingDataDefined:
        test_dataloader = DataLoader(
            test_data_for_torch,
            batch_size=1,
            pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
        )

    # Fetch the appropriate channel keys
    # Getting the channels for training and removing all the non numeric entries from the channels
    params = populate_channel_keys_in_params(validation_data_for_torch, params)

    # Calculate the weights here
    if params["weighted_loss"]:
        # Set up the dataloader for penalty calculation
        penalty_data = ImagesFromDataFrame(
            training_data,
            parameters=params,
            train=False,
        )
        penalty_loader = DataLoader(
            penalty_data,
            batch_size=1,
            shuffle=True,
            pin_memory=False,
        )

        params["weights"] = get_class_imbalance_weights(penalty_loader, params)
    else:
        params["weights"] = None

    # Fetch the optimizers
    optimizer = get_optimizer(
        optimizer_name=params["opt"],
        model=model,
        learning_rate=params["learning_rate"],
    )

    scheduler = get_scheduler(
        which_scheduler=params["scheduler"],
        optimizer=optimizer,
        batch_size=params["batch_size"],
        training_samples_size=len(train_dataloader.dataset),
        learning_rate=params["learning_rate"],
    )

    # Start training time here
    start_time = time.time()
    print("\n\n")
    # datetime object containing current date and time
    print("Initializing training at : ", get_date_time())

    # Setup a few loggers for tracking
    train_logger = Logger(
        logger_csv_filename=os.path.join(output_dir, "train_logs.csv"),
        metrics=params["metrics"],
    )
    valid_logger = Logger(
        logger_csv_filename=os.path.join(output_dir, "valid_logs.csv"),
        metrics=params["metrics"],
    )
    test_logger = Logger(
        logger_csv_filename=os.path.join(output_dir, "test_logs.csv"),
        metrics=params["metrics"],
    )
    train_logger.write_header(mode="train")
    valid_logger.write_header(mode="valid")
    test_logger.write_header(mode="valid")

    model, params["model"]["amp"], device = send_model_to_device(
        model, amp=params["model"]["amp"], device=params["device"], optimizer=optimizer
    )

    if "medcam" in params:
        model = medcam.inject(
            model,
            output_dir=os.path.join(
                output_dir, "attention_maps", params["medcam"]["backend"]
            ),
            backend=params["medcam"]["backend"],
            layer=params["medcam"]["layer"],
            save_maps=False,
            return_attention=True,
            enabled=False,
        )
        params["medcam_enabled"] = False

    # Setup a few variables for tracking
    best_loss = 1e7
    patience = 0

    print("Using device:", device, flush=True)

    first_model_saved = False

    # Iterate for number of epochs
    for epoch in range(epochs):

        # Printing times
        epoch_start_time = time.time()
        print("*" * 20)
        print("Starting Epoch : ", epoch)
        print("Epoch start time : ", get_date_time())

        epoch_train_loss, epoch_train_metric = train_network(
            model, train_dataloader, optimizer, params
        )
        epoch_valid_loss, epoch_valid_metric = validate_network(
            model, val_dataloader, scheduler, params, mode="validation"
        )

        patience += 1

        # Write the losses to a logger

        train_logger.write(epoch, epoch_train_loss, epoch_train_metric)
        valid_logger.write(epoch, epoch_valid_loss, epoch_valid_metric)

        if testingDataDefined:
            epoch_test_loss, epoch_test_metric = validate_network(
                model, test_dataloader, scheduler, params, mode="testing"
            )
            test_logger.write(epoch, epoch_test_loss, epoch_test_metric)

        print("Epoch end time : ", get_date_time())
        epoch_end_time = time.time()
        print(
            "Time taken for epoch : ",
            (epoch_end_time - epoch_start_time) / 60,
            " mins",
            flush=True,
        )

        # Start to check for loss
        if not (first_model_saved) or (epoch_valid_loss <= torch.tensor(best_loss)):
            best_loss = epoch_valid_loss
            best_train_idx = epoch
            patience = 0
            torch.save(
                {
                    "epoch": best_train_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                os.path.join(
                    output_dir, params["model"]["architecture"] + "_best.pth.tar"
                ),
            )
            first_model_saved = True

        if patience > params["patience"]:
            print(
                "Performance Metric has not improved for %d epochs, exiting training loop!"
                % (patience),
                flush=True,
            )
            break

    # End train time
    end_time = time.time()

    print(
        "Total time to finish Training : ",
        (end_time - start_time) / 60,
        " mins",
        flush=True,
    )
