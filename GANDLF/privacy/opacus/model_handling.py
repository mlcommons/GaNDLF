import collections.abc as abc
from functools import partial
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader
from typing import Union, Callable, Tuple
import copy

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def opacus_model_fix(model: torch.nn.Module, params: dict) -> torch.nn.Module:
    """
    Function to detect components of the model that are not compatible with Opacus differentially private training, and replacing with compatible components
    or raising an exception when a fix cannot be handled by Opacus.

    Args:
        model (torch.nn.Module): The model to be trained.
        params (dict): Training parameters.

    Returns:
        torch.nn.Module: Model, with potentially some components replaced with ones compatible with Opacus.
    """
    # use opacus to detect issues with model
    opacus_errors_detected = ModuleValidator.validate(model, strict=False)

    if not params["differential_privacy"]["allow_opacus_model_fix"]:
        assert (
            opacus_errors_detected == []
        ), f"Training parameters are set to not allow Opacus to try to fix incompatible model components, and the following issues were detected: {opacus_errors_detected}"
    elif opacus_errors_detected != []:
        print(
            f"Allowing Opacus to try and patch the model due to the following issues: {opacus_errors_detected}"
        )
        print()
        model = ModuleValidator.fix(model)
        # If the fix did not work, raise an exception
        ModuleValidator.validate(model, strict=True)
    return model


def prep_for_opacus_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    params: dict,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader, PrivacyEngine]:
    """
    Function to prepare the model, optimizer, and dataloader for differentially private training using Opacus.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        train_dataloader (DataLoader): The dataloader for the training data.
        params (dict): Training parameters.

    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader, PrivacyEngine]: Model, optimizer, dataloader, and privacy engine.
    """

    privacy_engine = PrivacyEngine(
        accountant=params["differential_privacy"]["accountant"],
        secure_mode=params["differential_privacy"]["secure_mode"],
    )

    if not "epsilon" in params["differential_privacy"]:
        model, optimizer, train_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            noise_multiplier=params["differential_privacy"]["noise_multiplier"],
            max_grad_norm=params["differential_privacy"]["max_grad_norm"],
        )
    else:
        (model, optimizer, train_dataloader) = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            max_grad_norm=params["differential_privacy"]["max_grad_norm"],
            epochs=params["num_epochs"],
            target_epsilon=params["differential_privacy"]["epsilon"],
            target_delta=params["differential_privacy"]["delta"],
        )
    return model, optimizer, train_dataloader, privacy_engine


def build_empty_batch_value(
    sample: Union[torch.Tensor, np.ndarray, abc.Mapping, abc.Sequence, int, float, str]
):
    """
    Build an empty batch value from a sample. This function is used to create a placeholder for empty batches in an iteration. Inspired from https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py#L108. The key difference is that pytorch `collate` has to traverse batch of objects AND unite its fields to lists, while this function traverse a single item AND creates an "empty" version of the batch.

    Args:
        sample (Union[torch.Tensor, np.ndarray, abc.Mapping, abc.Sequence, int, float, str]): A sample from the dataset.

    Raises:
        TypeError: If the data type is not supported.

    Returns:
        Union[torch.Tensor, np.ndarray, abc.Mapping, abc.Sequence, int, float, str]: An empty batch value.
    """
    if isinstance(sample, torch.Tensor):
        # Create an empty tensor with the same shape except for the zeroed batch dimension.
        return torch.empty((0,) + sample.shape)
    elif isinstance(sample, np.ndarray):
        # Create an empty tensor from a numpy array, also with the zeroed batch dimension.
        return torch.empty((0,) + sample.shape, dtype=torch.from_numpy(sample).dtype)
    elif isinstance(sample, abc.Mapping):
        # Recursively handle dictionary-like objects.
        return {key: build_empty_batch_value(value) for key, value in sample.items()}
    elif isinstance(sample, tuple) and hasattr(sample, "_fields"):  # namedtuple
        return type(sample)(*(build_empty_batch_value(item) for item in sample))
    elif isinstance(sample, abc.Sequence) and not isinstance(sample, str):
        # Handle lists and tuples, but exclude strings.
        return [build_empty_batch_value(item) for item in sample]
    elif isinstance(sample, (int, float, str)):
        # Return an empty list for basic data types.
        return []
    else:
        raise TypeError(f"Unsupported data type: {type(sample)}")


def empty_collate(
    item_example: Union[
        torch.Tensor, np.ndarray, abc.Mapping, abc.Sequence, int, float, str
    ]
) -> Callable:
    """
    Creates a new collate function that behave same as default pytorch one,
    but can process the empty batches.

    Args:
        item_example (Union[torch.Tensor, np.ndarray, abc.Mapping, abc.Sequence, int, float, str]): An example item from the dataset.

    Returns:
        Callable: function that should replace dataloader collate: `dataloader.collate_fn = empty_collate(...)`
    """

    def custom_collate(batch, _empty_batch_value):
        if len(batch) > 0:
            return default_collate(batch)  # default behavior
        else:
            return copy.copy(_empty_batch_value)

    empty_batch_value = build_empty_batch_value(item_example)

    return partial(custom_collate, _empty_batch_value=empty_batch_value)
