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


def create_pytorch_objects(parameters, train_csv, val_csv, device):
    """
    _summary_

    Args:
        parameters (_type_): _description_
        train_csv (_type_): _description_
        val_csv (_type_): _description_
        device (_type_): _description_

    Returns:
        model (_type_): _description_
        optimizer (_type_): _description_
        train_loader (_type_): _description_
        val_loader (_type_): _description_
        scheduler (_type_): _description_
    """
    # populate the data frames
    parameters["training_data"], headers_train = parseTrainingCSV(train_csv, train=True)
    parameters = populate_header_in_parameters(parameters, headers_train)
    parameters["validation_data"], _ = parseTrainingCSV(val_csv, train=False)

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

    # get the train loader
    train_loader = get_train_loader(parameters)
    parameters["training_samples_size"] = len(train_loader)
    # get the validation loader
    val_loader = get_validation_loader(parameters)

    if not ("step_size" in parameters["scheduler"]):
        parameters["scheduler"]["step_size"] = (
            parameters["training_samples_size"] / parameters["learning_rate"]
        )

    scheduler = get_scheduler(parameters)

    # these keys contain generators, and are not needed beyond this point in params
    generator_keys_to_remove = ["optimizer_object", "model_parameters"]
    for key in generator_keys_to_remove:
        parameters.pop(key, None)

    validation_data_for_torch = ImagesFromDataFrame(
        parameters["validation_data"], parameters, train=False, loader_type="validation"
    )
    # Fetch the appropriate channel keys
    # Getting the channels for training and removing all the non numeric entries from the channels
    parameters = populate_channel_keys_in_params(validation_data_for_torch, parameters)

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

    return model, optimizer, train_loader, val_loader, scheduler, parameters
