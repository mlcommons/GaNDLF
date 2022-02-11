from .ImagesFromDataFrame import ImagesFromDataFrame
from torch.utils.data import DataLoader

def get_train_loader(params):
    """
    Get the training data loader.

    Args:
        params (dict): Dictionary of parameters.

    Returns:
        torch.utils.data.DataLoader: The training loader.
    """
    return DataLoader(
        ImagesFromDataFrame(params["training_data"], params, train=True, loader_type="train"),
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
    return DataLoader(
        ImagesFromDataFrame(params["validation_data"], params, train=False, loader_type="validation"),
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
    return DataLoader(
        ImagesFromDataFrame(params["testing_data"], params, train=False, loader_type="testing"),
        batch_size=1,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )