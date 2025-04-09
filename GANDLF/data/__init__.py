from torch.utils.data import DataLoader

from .ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.utils.write_parse import get_dataframe
from GANDLF.utils import populate_channel_keys_in_params


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
            get_dataframe(params["training_data"]),
            params,
            train=True,
            loader_type="train",
        ),
        batch_size=params["batch_size"],
        shuffle=True,
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
    queue_from_dataframe = ImagesFromDataFrame(
        get_dataframe(params["validation_data"]),
        params,
        train=False,
        loader_type="validation",
    )
    # Fetch the appropriate channel keys
    # Getting the channels for training and removing all the non numeric entries from the channels
    params = populate_channel_keys_in_params(queue_from_dataframe, params)

    return DataLoader(
        queue_from_dataframe,
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

    queue_from_dataframe = ImagesFromDataFrame(
        get_dataframe(params["testing_data"]),
        params,
        train=False,
        loader_type="testing",
    )
    if not ("channel_keys" in params):
        params = populate_channel_keys_in_params(queue_from_dataframe, params)
    return DataLoader(
        queue_from_dataframe,
        batch_size=1,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )
