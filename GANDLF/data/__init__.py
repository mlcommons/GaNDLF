from torch.utils.data import DataLoader

from .ImagesFromDataFrame import ImagesFromDataFrame
from ..utils import get_dataframe


def get_train_loader(params):
    """
    Get the training data loader.

    Args:
        params (dict): Dictionary of parameters.

    Returns:
        torch.utils.data.DataLoader: The training loader.
    """

    return DataLoader(
        ImagesFromDataFrame(
            get_dataframe(params["training_data"]), params, train=True, loader_type="train"
        ),
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )


def get_penalty_loader(params):
    """
    Get the penalty data loader.

    Args:
        params (dict): Dictionary of parameters.

    Returns:
        torch.utils.data.DataLoader: The penalty loader.
    """
    return DataLoader(
        ImagesFromDataFrame(
            get_dataframe(params["training_data"]), params, train=True, loader_type="penalty"
        ),
        batch_size=1,
        shuffle=False,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )


def get_validation_loader(params):
    """
    Get the validation data loader.

    Args:
        params (dict): Dictionary of parameters.

    Returns:
        torch.utils.data.DataLoader: The validation loader.
    """
    return DataLoader(
        ImagesFromDataFrame(
            get_dataframe(params["validation_data"]), params, train=False, loader_type="validation"
        ),
        batch_size=1,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )


def get_testing_loader(params):
    """
    Get the testing data loader.

    Args:
        params (dict): Dictionary of parameters.

    Returns:
        torch.utils.data.DataLoader: The testing loader.
    """
    if params["testing_data"] is None:
        return None
    else:
        return DataLoader(
            ImagesFromDataFrame(
                get_dataframe(params["testing_data"]), params, train=False, loader_type="testing"
            ),
            batch_size=1,
            pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
        )
