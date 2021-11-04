import os
from pathlib import Path

from GANDLF.utils import (
    get_date_time,
    send_model_to_device,
    populate_channel_keys_in_params,
    get_class_imbalance_weights,
    populate_header_in_parameters, 
    parseTrainingCSV
)

from GANDLF.models import global_models_dict
from GANDLF.schedulers import global_schedulers_dict
from GANDLF.optimizers import global_optimizer_dict

from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame

import pickle
import torch
import pandas as pd
from torch.utils.data import DataLoader

def generate_data_loader(DATA_DIR, N_FOLD, is_train=True):
    train_mode=True
    train_csv=os.path.join(DATA_DIR, "csv_files/", N_FOLD,  "data_training.csv")
    validation_csv=os.path.join(DATA_DIR, "csv_files/", N_FOLD,  "data_validation.csv")
    data_train, headers_train = parseTrainingCSV(train_csv, train=train_mode)

    data_validation, headers_validation = parseTrainingCSV(validation_csv, train=train_mode)

    with open(os.path.join(DATA_DIR, 'parameters.pkl'), 'rb') as f:
        parameters = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    parameters = populate_header_in_parameters(parameters, headers_train)
    parameters["output_dir"] = Path(os.path.join(DATA_DIR, "patch_data", N_FOLD))
    parameters["device"] = device

    model = global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    # Set up the dataloaders
    if is_train:
        training_data_for_torch = ImagesFromDataFrame(data_train, parameters, train=True)
        train_dataloader = DataLoader(
            training_data_for_torch,
            batch_size=parameters["batch_size"],
            shuffle=True,
            pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
        )
    else:
        training_data_for_torch = ImagesFromDataFrame(data_train, parameters, train=False)
        train_dataloader = DataLoader(
            training_data_for_torch,
            batch_size=1,
            shuffle=True,
            pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
        )

    validation_data_for_torch = ImagesFromDataFrame(data_validation, parameters, train=False)
    parameters["training_samples_size"] = len(train_dataloader.dataset)

    testingDataDefined = False

    val_dataloader = DataLoader(
        validation_data_for_torch,
        batch_size=1,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )

    parameters = populate_channel_keys_in_params(validation_data_for_torch, parameters)

    if parameters["weighted_loss"]:
        # Set up the dataloader for penalty calculation
        penalty_data = ImagesFromDataFrame(
            data_train,
            parameters=parameters,
            train=False,
        )
        penalty_loader = DataLoader(
            penalty_data,
            batch_size=1,
            shuffle=True,
            pin_memory=False,
        )

        parameters["weights"], parameters["class_weights"] = get_class_imbalance_weights(
            penalty_loader, parameters
        )
    else:
        parameters["weights"], parameters["class_weights"] = None, None

    
    parameters["model_parameters"] = model.parameters()
    optimizer = global_optimizer_dict[parameters["optimizer"]["type"]](parameters)
    parameters["optimizer_object"] = optimizer
    if not ("step_size" in parameters["scheduler"]):
            parameters["scheduler"]["step_size"] = (
                parameters["training_samples_size"] / parameters["learning_rate"]
            )

    scheduler = global_schedulers_dict[parameters["scheduler"]["type"]](parameters)

    return model, parameters, train_dataloader, val_dataloader, scheduler, optimizer
