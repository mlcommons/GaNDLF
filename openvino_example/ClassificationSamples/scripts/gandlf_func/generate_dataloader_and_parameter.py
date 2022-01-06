import os
from pathlib import Path
from GANDLF.models import global_models_dict

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


def generate_data_loader(data_csv, parameters, out_dir, verbose=False, train_mode=False):
    data_full, headers = parseTrainingCSV(data_csv, train=train_mode)
    parameters = populate_header_in_parameters(parameters, headers)

    inferenceDataForTorch = ImagesFromDataFrame(data_full, parameters, train=False)
    parameters = populate_channel_keys_in_params(inferenceDataForTorch, parameters)
    parameters['device'] = 'cpu'
    parameters["model"]["amp"] = None
    parameters["weights"], parameters["class_weights"] = None, None
    parameters["output_dir"] = out_dir
    parameters["verbose"] = False

    print(parameters)

    model = global_models_dict[parameters["model"]
                           ["architecture"]](parameters=parameters)

    infer_dataloader = torch.utils.data.DataLoader(
        inferenceDataForTorch,
        batch_size=parameters["batch_size"],
        shuffle=False,
        pin_memory=True  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )

    return model, infer_dataloader, parameters
