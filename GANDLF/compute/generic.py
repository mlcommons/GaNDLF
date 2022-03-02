from GANDLF.models import get_model
from GANDLF.schedulers import get_scheduler
from GANDLF.optimizers import get_optimizer
from GANDLF.data import (
    get_train_loader,
    get_validation_loader,
    get_penalty_loader,
    ImagesFromDataFrame,
)
from GANDLF.utils import (
    populate_channel_keys_in_params,
    populate_header_in_parameters,
    parseTrainingCSV,
    send_model_to_device,
    get_class_imbalance_weights,
)


def create_pytorch_objects(parameters, train_csv=None, val_csv=None, device="cpu"):
    """
    This function creates all the PyTorch objects needed for training.

    Args:
        parameters (dict): The parameters dictionary.
        train_csv (str): The path to the training CSV file.
        val_csv (str): The path to the validation CSV file.
        device (str): The device to perform computations on.

    Returns:
        model (torch.nn.Module): The model to use for training.
        optimizer (Optimizer): The optimizer to use for training.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        scheduler (object): The scheduler to use for training.
        parameters (dict): The updated parameters dictionary.
    """
    # initialize train and val loaders
    train_loader, val_loader = None, None

    if train_csv is not None:
        # populate the data frames
        parameters["training_data"], headers_train = parseTrainingCSV(
            train_csv, train=True
        )
        parameters = populate_header_in_parameters(parameters, headers_train)
        # get the train loader
        train_loader = get_train_loader(parameters)
        parameters["training_samples_size"] = len(train_loader)

        # Calculate the weights here
        if parameters["weighted_loss"]:
            print("Calculating weights for loss")
            penalty_loader = get_penalty_loader(parameters)

            (
                parameters["weights"],
                parameters["class_weights"],
            ) = get_class_imbalance_weights(penalty_loader, parameters)
            del penalty_loader
        else:
            parameters["weights"], parameters["class_weights"] = None, None

    if val_csv is not None:
        parameters["validation_data"], _ = parseTrainingCSV(val_csv, train=False)
        # get the validation loader
        val_loader = get_validation_loader(parameters)

        validation_data_for_torch = ImagesFromDataFrame(
            parameters["validation_data"],
            parameters,
            train=False,
            loader_type="populating_headers",
        )
        # Fetch the appropriate channel keys
        # Getting the channels for training and removing all the non numeric entries from the channels
        parameters = populate_channel_keys_in_params(
            validation_data_for_torch, parameters
        )

    # get the model
    model = get_model(parameters)
    parameters["model_parameters"] = model.parameters()

    # get the optimizer
    optimizer = get_optimizer(parameters)
    parameters["optimizer_object"] = optimizer

    # send model to correct device
    model, parameters["model"]["amp"], parameters["device"] = send_model_to_device(
        model, amp=parameters["model"]["amp"], device=device, optimizer=optimizer
    )

    if not ("step_size" in parameters["scheduler"]):
        parameters["scheduler"]["step_size"] = (
            parameters["training_samples_size"] / parameters["learning_rate"]
        )

    scheduler = get_scheduler(parameters)

    # these keys contain generators, and are not needed beyond this point in params
    generator_keys_to_remove = ["optimizer_object", "model_parameters"]
    for key in generator_keys_to_remove:
        parameters.pop(key, None)

    return model, optimizer, train_loader, val_loader, scheduler, parameters
