#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 21:45:06 2021

@author: siddhesh
"""
import os
import torch
import time
import torchio
from torch.utils.data import DataLoader
from GANDLF.logger import Logger
from GANDLF.losses import fetch_loss_function
from GANDLF.metrics import fetch_metric
from GANDLF.parameterParsing import get_model, get_optimizer
from GANDLF.utils import get_date_time, send_model_to_device
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame

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
        the parameters passed by the user yaml

    Returns
    -------
    loss : torch.Tensor
        The computed loss from the label and the output
    metric_output : torch.Tensor
        The computed metric from the label and the output

    """
    if params["model"]["amp"]:
        with torch.cuda.amp.autocast():
            output = model(image)
    else:
        output = model(image)
    loss_function = fetch_loss_function(params["loss"])  # Write a fetch_loss_function
    loss = loss_function(output, label, params)
    metric_output = {}
    # Metrics should be a list
    for metric in params["metrics"]:
        metric_function = fetch_metric(metric)  # Write a fetch_metric
        metric_output[metric] = metric_function(output, label, params)
    return loss, metric_output


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

    for metric in params["metrics"].keys():
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
        image = torch.cat(
            [subject[key][torchio.DATA] for key in params["channel_keys"]], dim=1
        ).to(params["device"])
        if params["value_keys"] is not None:
            label = torch.cat([subject[key] for key in params["value_keys"]], dim=0)
            label = torch.reshape(
                subject[params["value_keys"][0]], (params["batch_size"], 1)
            )
        else:
            label = subject["label"][torchio.DATA]
        label = label.to(params["device"])
        loss, calculated_metrics = step(model, image, label, batch_idx, params)
        if params["model"]["amp"]:
            with torch.cuda.amp.autocast():
                if not torch.isnan(
                    loss
                ):  # if loss is nan, dont backprop and dont step optimizer
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Non network training related
        total_epoch_train_loss += loss
        for metric in calculated_metrics.keys():
            total_epoch_train_metric[metric] += calculated_metrics[metric]

        # For printing information at halftime during an epoch
        if batch_idx % (len(train_dataloader) // 2) == 0:
            print("Epoch Average Train loss : ", total_epoch_train_loss / batch_idx)
            for metric in params["metrics"].keys():
                print(
                    "Epoch Average Train " + metric + " : ",
                    total_epoch_train_metric[metric] / batch_idx,
                )

    average_epoch_train_loss = total_epoch_train_loss / len(train_dataloader)
    for metric in params["metrics"].keys():
        average_epoch_train_metric[metric] = total_epoch_train_metric[metric] / len(
            train_dataloader
        )

    return average_epoch_train_loss, average_epoch_train_metric


def validate_network(model, valid_dataloader, params):
    """
    Function to validate a network for a single epoch

    Parameters
    ----------
    model : torch.model
        The model to process the input image with, it should support appropriate dimensions.
    valid_dataloader : torch.DataLoader
        The dataloader for the validation epoch
    params : dict
        the parameters passed by the user yaml

    Returns
    -------
    average_epoch_valid_loss : float
        Validation loss for the current epoch
    average_epoch_valid_metric : dict
        Validation metrics for the current epoch

    """
    print("*" * 20)
    print("Starting Epoch : ")
    print("*" * 20)
    # Initialize a few things
    total_epoch_valid_loss = 0
    total_epoch_valid_metric = {}
    average_epoch_valid_metric = {}

    for metric in params["metrics"].keys():
        total_epoch_valid_metric[metric] = 0

    # automatic mixed precision - https://pytorch.org/docs/stable/amp.html
    if params["model"]["amp"]:
        print("Using Automatic mixed precision", flush=True)

    # Fetch the optimizer

    # Set the model to valid
    model.eval()
    for batch_idx, (subject) in enumerate(valid_dataloader):
        image = torch.cat(
            [subject[key][torchio.DATA] for key in params["channel_keys"]], dim=1
        ).to(params["device"])
        if params["value_keys"] is not None:
            label = torch.cat([subject[key] for key in params["value_keys"]], dim=0)
            label = torch.reshape(
                subject[params["value_keys"][0]], (params["batch_size"], 1)
            )
        else:
            label = subject["label"][torchio.DATA]
        label = label.to(params["device"])
        loss, calculated_metrics = step(model, image, label, batch_idx, params)

        # Non network validing related
        total_epoch_valid_loss += loss
        for metric in calculated_metrics.keys():
            total_epoch_valid_metric[metric] += calculated_metrics[metric]

        # For printing information at halftime during an epoch
        if batch_idx % (len(valid_dataloader) // 2) == 0:
            print(
                "Epoch Average Validation loss : ", total_epoch_valid_loss / batch_idx
            )
            for metric in params["metrics"].keys():
                print(
                    "Epoch Validation " + metric + " : ",
                    total_epoch_valid_metric[metric] / len(valid_dataloader),
                )

    average_epoch_valid_loss = total_epoch_valid_loss / len(valid_dataloader)
    for metric in params["metrics"].keys():
        average_epoch_valid_metric[metric] = total_epoch_valid_metric[metric] / len(
            valid_dataloader
        )

    return average_epoch_valid_loss, average_epoch_valid_metric


def training_loop(
    training_data,
    validation_data,
    headers,
    device,
    params,
    output_dir,
    testing_data=None,
):

    # Some autodetermined factors
    num_classes = len(params["model"]["class_list"])
    params["model"]["num_classes"] = num_classes
    params["headers"] = headers
    epochs = params["num_epochs"]
    loss = params["loss"]
    metrics = params["metrics"]
    params["device"] = device
    device = params["device"]

    # Fetch the model according to params mentioned in the configuration file
    model = get_model(
        modelname=params["model"]["architecture"],
        num_dimension=params["model"]["dimension"],
        num_channels=params["model"]["num_channels"],
        num_classes=params["model"]["num_classes"],
        base_filters=params["model"]["base_filters"],
        final_convolution_layer=params["model"]["final_layer"],
        patch_size=params["patch_size"],
        batch_size=params["batch_size"],
    )

    # Set up the dataloaders
    training_data_for_torch = ImagesFromDataFrame(
        training_data,
        patch_size=params["patch_size"],
        headers=params["headers"],
        q_max_length=params["q_max_length"],
        q_samples_per_volume=params["q_samples_per_volume"],
        q_num_workers=params["q_num_workers"],
        q_verbose=params["q_verbose"],
        sampler=params["patch_sampler"],
        augmentations=params["data_augmentations"],
        preprocessing=params["data_preprocessing"],
        in_memory=params["in_memory"],
        train=True,
    )

    validation_data_for_torch = ImagesFromDataFrame(
        validation_data,
        patch_size=params["patch_size"],
        headers=params["headers"],
        q_max_length=params["q_max_length"],
        q_samples_per_volume=params["q_samples_per_volume"],
        q_num_workers=params["q_num_workers"],
        q_verbose=params["q_verbose"],
        sampler=params["patch_sampler"],
        augmentations=params["data_augmentations"],
        preprocessing=params["data_preprocessing"],
        in_memory=params["in_memory"],
        train=False,
    )

    testingDataDefined = True
    if testing_data is None:
        print(
            "No testing data is defined, using validation data for those metrics",
            flush=True,
        )
        testing_data = validation_data
        testingDataDefined = False

    test_data_for_torch = ImagesFromDataFrame(
        testing_data,
        patch_size=params["patch_size"],
        headers=params["headers"],
        q_max_length=params["q_max_length"],
        q_samples_per_volume=params["q_samples_per_volume"],
        q_num_workers=params["q_num_workers"],
        q_verbose=params["q_verbose"],
        sampler=params["patch_sampler"],
        augmentations=params["data_augmentations"],
        preprocessing=params["data_preprocessing"],
        in_memory=params["in_memory"],
        train=False,
    )

    train_dataloader = DataLoader(
        training_data_for_torch,
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=params["in_memory"],
    )

    val_dataloader = DataLoader(
        validation_data_for_torch, batch_size=1, pin_memory=params["in_memory"]
    )

    test_dataloader = DataLoader(test_data_for_torch, batch_size=1)

    # Fetch the optimizers
    optimizer = get_optimizer(
        optimizer_name=params["optimizer"],
        model_parameters=model,
        learning_rate=params["learning_rate"],
    )

    # Fetch the appropriate channel keys
    # Getting the channels for training and removing all the non numeric entries from the channels
    batch = next(
        iter(val_dataloader)
    )  # using train_loader makes this slower as train loader contains full augmentations
    all_keys = list(batch.keys())
    channel_keys = []
    value_keys = []
    print("Channel Keys : ", all_keys)
    for item in all_keys:
        if item.isnumeric():
            channel_keys.append(item)
        elif "value" in item:
            value_keys.append(item)
    params["channel_keys"] = channel_keys
    params["value_keys"] = value_keys

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
    train_logger.write_header(mode="train")
    valid_logger.write_header(mode="valid")

    model, amp, device = send_model_to_device(
        model, amp=params["model"]["amp"], device=params["device"], optimizer=optimizer
    )

    # Setup a few variables for tracking
    best_loss = 1e7
    patience = 0

    # Iterate for number of epochs
    for epoch in range(epochs):

        print("Using device:", device, flush=True)

        # Printing times
        epoch_start_time = time.time()
        print("*" * 20)
        print("Starting Epoch : ", epoch)
        print("Epoch start time : ", get_date_time())

        epoch_train_loss, epoch_train_metric = train_network(
            epoch, model, train_dataloader, optimizer, loss, metrics, params
        )
        epoch_valid_loss, epoch_valid_metric = validate_network(
            epoch, model, val_dataloader, loss, metrics, params
        )
        patience += 1

        # Write the losses to a logger
        train_logger.write(epoch, epoch_train_loss, epoch_train_metric)
        valid_logger.write(epoch, epoch_valid_loss, epoch_valid_metric)

        print("Epoch end time : ", get_date_time())
        epoch_end_time = time.time()
        print(
            "Time taken for epoch : ",
            (epoch_end_time - epoch_start_time) / 60,
            " mins",
            flush=True,
        )

        # Start to check for loss
        if epoch_valid_loss <= best_loss:
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
                os.path.join(output_dir, params['model']['architecture'] + "_best.pth.tar"),
            )

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
