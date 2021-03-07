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
from GANDLF.losses import fetch_loss_function
from GANDLF.metrics import fetch_metric
from GANDLF.parameterParsing import get_model, get_optimizer
from GANDLF.utils import get_date_time
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame

os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = "1"  # hides torchio citation request

# Reminder, the scaling factor should go to the metric MSE, and all should support a scaling factor, right?


def step(model, image, label, params):
    if params["amp"]:
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

    average_epoch_train_loss = total_epoch_train_loss / batch_idx
    for metric in params["metrics"].keys():
        average_epoch_train_metric[metric] = (
            total_epoch_train_metric[metric] / batch_idx
        )

    return average_epoch_train_loss, average_epoch_train_metric


def validate_network(epoch_count, model, valid_dataloader, params):

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
            print("Epoch Average Validation loss : ", total_epoch_valid_loss / batch_idx)
            for metric in params["metrics"].keys():
                print(
                    "Epoch Validation " + metric + " : ",
                    total_epoch_valid_metric[metric] / batch_idx,
                )

    average_epoch_valid_loss = total_epoch_valid_loss / batch_idx
    for metric in params["metrics"].keys():
        average_epoch_valid_metric[metric] = (
            total_epoch_valid_metric[metric] / batch_idx
        )

    return average_epoch_valid_loss, average_epoch_valid_metric


def test_network(epoch_count, model, valid_dataloader, params):

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
            print("Epoch average loss : ", total_epoch_valid_loss / batch_idx)
            for metric in params["metrics"].keys():
                print(
                    "Epoch Average " + metric + " : ",
                    total_epoch_valid_metric[metric] / batch_idx,
                )

    average_epoch_valid_loss = total_epoch_valid_loss / batch_idx
    for metric in params["metrics"].keys():
        average_epoch_valid_metric[metric] = (
            total_epoch_valid_metric[metric] / batch_idx
        )

    return average_epoch_valid_loss, average_epoch_valid_metric


def training_loop(
    training_data,
    validation_data,
    headers,
    device,
    params,
    outputDir,
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
    trainingDataForTorch = ImagesFromDataFrame(
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

    validationDataForTorch = ImagesFromDataFrame(
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

    testDataForTorch = ImagesFromDataFrame(
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
        trainingDataForTorch,
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=params["in_memory"],
    )
    val_dataloader = DataLoader(
        validationDataForTorch, batch_size=1, pin_memory=params["in_memory"]
    )
    test_dataloader = DataLoader(testDataForTorch, batch_size=1)

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

    # Iterate for number of epochs
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print("*" * 20)
        print("Starting Epoch : ", epoch)
        print("Epoch start time : ", get_date_time())
        train_network(epoch, model, train_dataloader, optimizer, loss, metrics, params)
        validate_network(epoch, model, val_dataloader, loss, metrics, params)
        print("Epoch end time : ", get_date_time())
        epoch_end_time = time.time()
        print(
            "Time take for epoch : ",
            (epoch_end_time - epoch_start_time) / 60,
            " mins",
            flush=True,
        )

    # End train time
    end_time = time.time()

    print(
        "Total time to finish Training : ",
        (end_time - start_time) / 60,
        " mins",
        flush=True,
    )

    test_network(model, test_dataloader, params)
    # Do rest of the stuff later
