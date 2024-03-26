from typing import Optional, Tuple
from pandas.util import hash_pandas_object
import torch
from torch.utils.data import DataLoader

from GANDLF.models import get_model
from GANDLF.schedulers import get_scheduler
from GANDLF.optimizers import get_optimizer
from GANDLF.data import get_train_loader, get_validation_loader
from GANDLF.utils import (
    populate_header_in_parameters,
    parseTrainingCSV,
    send_model_to_device,
    get_class_imbalance_weights,
)


def create_pytorch_objects(
    parameters: dict,
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None,
    device: Optional[str] = "cpu",
) -> Tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    DataLoader,
    DataLoader,
    torch.optim.lr_scheduler.LRScheduler,
    dict,
]:
    """
    This function creates the PyTorch objects needed for training and validation.

    Args:
        parameters (dict): The parameters for the model and training.
        train_csv (Optional[str], optional): The path to the training CSV file. Defaults to None.
        val_csv (Optional[str], optional): The path to the validation CSV file. Defaults to None.
        device (Optional[str], optional): The device to use for training. Defaults to "cpu".

    Returns:
        Tuple[ torch.nn.Module, torch.optim.Optimizer, DataLoader, DataLoader, torch.optim.lr_scheduler.LRScheduler, dict, ]: The model, optimizer, train loader, validation loader, scheduler, and parameters.
    """
    # initialize train and val loaders
    train_loader, val_loader = None, None
    headers_to_populate_train, headers_to_populate_val = None, None

    if train_csv is not None:
        # populate the data frames
        parameters["training_data"], headers_to_populate_train = parseTrainingCSV(
            train_csv, train=True
        )
        parameters = populate_header_in_parameters(
            parameters, headers_to_populate_train
        )

        # Calculate the weights here
        (
            parameters["penalty_weights"],
            parameters["sampling_weights"],
            parameters["class_weights"],
        ) = get_class_imbalance_weights(parameters["training_data"], parameters)

        print("Penalty weights : ", parameters["penalty_weights"])
        print("Sampling weights: ", parameters["sampling_weights"])
        print("Class weights   : ", parameters["class_weights"])

        # get the train loader
        train_loader = get_train_loader(parameters)
        parameters["training_samples_size"] = len(train_loader)
        # get the hash of the training data for reproducibility
        parameters["training_data_hash"] = hash_pandas_object(
            parameters["training_data"]
        ).sum()

    if val_csv is not None:
        parameters["validation_data"], headers_to_populate_val = parseTrainingCSV(
            val_csv, train=False
        )
        if headers_to_populate_train is None:
            parameters = populate_header_in_parameters(
                parameters, headers_to_populate_val
            )
        # get the validation loader
        val_loader = get_validation_loader(parameters)

    # get the model
    model = get_model(parameters)
    parameters["model_parameters"] = model.parameters()

    # get the optimizer
    optimizer = get_optimizer(parameters)
    parameters["optimizer_object"] = optimizer

    # send model to correct device
    (
        model,
        parameters["model"]["amp"],
        parameters["device"],
        parameters["device_id"],
    ) = send_model_to_device(
        model, amp=parameters["model"]["amp"], device=device, optimizer=optimizer
    )

    # only need to create scheduler if training
    if train_csv is not None:
        if not ("step_size" in parameters["scheduler"]):
            parameters["scheduler"]["step_size"] = (
                parameters["training_samples_size"] / parameters["learning_rate"]
            )

        scheduler = get_scheduler(parameters)

    else:
        scheduler = None

    # these keys contain generators, and are not needed beyond this point in params
    generator_keys_to_remove = ["optimizer_object", "model_parameters"]
    for key in generator_keys_to_remove:
        parameters.pop(key, None)

    return model, optimizer, train_loader, val_loader, scheduler, parameters
