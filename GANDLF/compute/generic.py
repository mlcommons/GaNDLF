from typing import Optional, Tuple
from pandas.util import hash_pandas_object
import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from dataclasses import dataclass
from GANDLF.models import get_model
from GANDLF.schedulers import get_scheduler
from GANDLF.optimizers import get_optimizer
from GANDLF.data import get_train_loader, get_validation_loader, ImagesFromDataFrame
from GANDLF.utils import (
    populate_header_in_parameters,
    populate_channel_keys_in_params,
    parseTrainingCSV,
    send_model_to_device,
    get_class_imbalance_weights,
)
from GANDLF.utils.write_parse import get_dataframe
from torchio import SubjectsDataset, Queue
from typing import Union


@dataclass
class AbstractSubsetDataParser(ABC):
    """
    Interface for subset data parsers, needed to separate the dataset creation
    from construction of the dataloaders.
    """

    subset_csv_path: str
    parameters_dict: dict

    @abstractmethod
    def create_subset_dataset(self) -> Union[SubjectsDataset, Queue]:
        """
        Method to create the subset dataset based on the subset CSV file
        and the parameters dict.

        Returns:
            Union[SubjectsDataset, Queue]: The subset dataset.
        """
        pass

    def get_params_extended_with_subset_data(self) -> dict:
        """
        Trick to get around the fact that parameters dict need to be modified
        during this parsing procedure. This method should be called after
        create_subset_dataset(), as this method will populate the parameters
        dict with the headers from the subset data.
        """
        return self.parameters_dict


class TrainingSubsetDataParser(AbstractSubsetDataParser):
    def create_subset_dataset(self) -> Union[SubjectsDataset, Queue]:
        (
            self.parameters_dict["training_data"],
            headers_to_populate_train,
        ) = parseTrainingCSV(self.subset_csv_path, train=True)

        self.parameters_dict = populate_header_in_parameters(
            self.parameters_dict, headers_to_populate_train
        )

        (
            self.parameters_dict["penalty_weights"],
            self.parameters_dict["sampling_weights"],
            self.parameters_dict["class_weights"],
        ) = get_class_imbalance_weights(
            self.parameters_dict["training_data"], self.parameters_dict
        )

        print("Penalty weights : ", self.parameters_dict["penalty_weights"])
        print("Sampling weights: ", self.parameters_dict["sampling_weights"])
        print("Class weights   : ", self.parameters_dict["class_weights"])

        return ImagesFromDataFrame(
            get_dataframe(self.parameters_dict["training_data"]),
            self.parameters_dict,
            train=True,
            loader_type="train",
        )


class ValidationSubsetDataParser(AbstractSubsetDataParser):
    def create_subset_dataset(self) -> Union[SubjectsDataset, Queue]:
        (self.parameters_dict["validation_data"], _) = parseTrainingCSV(
            self.subset_csv_path, train=False
        )
        validation_dataset = ImagesFromDataFrame(
            get_dataframe(self.parameters_dict["validation_data"]),
            self.parameters_dict,
            train=False,
            loader_type="validation",
        )
        self.parameters_dict = populate_channel_keys_in_params(
            validation_dataset, self.parameters_dict
        )
        return validation_dataset


class TestSubsetDataParser(AbstractSubsetDataParser):
    def create_subset_dataset(self) -> Union[SubjectsDataset, Queue]:
        testing_dataset = ImagesFromDataFrame(
            get_dataframe(self.subset_csv_path),
            self.parameters_dict,
            train=False,
            loader_type="testing",
        )
        if not ("channel_keys" in self.parameters_dict):
            self.parameters_dict = populate_channel_keys_in_params(
                testing_dataset, self.parameters_dict
            )
        return testing_dataset


class InferenceSubsetDataParserRadiology(TestSubsetDataParser):
    """Simple wrapper for name coherency, functionally this is the same as TestSubsetDataParser"""

    pass


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
