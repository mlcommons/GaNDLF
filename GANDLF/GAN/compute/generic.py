import torch
from pandas.util import hash_pandas_object
from GANDLF.models import get_model
from GANDLF.GAN.schedulers import get_scheduler_gan
from GANDLF.GAN.optimizers import get_optimizers_gan
from GANDLF.data import (
    get_train_loader,
    get_validation_loader,
)
from GANDLF.utils import (
    populate_header_in_parameters,
    parseTrainingCSV,
    send_model_to_device,
    get_class_imbalance_weights,
)


def generate_latent_vector(
    batch_size: int, latent_vector_size: int, dimension: int, device: str
) -> torch.Tensor:
    """Creates a latent vector of given size and adjusts the dimensions
    according to the dimension parameter (for 2D or 3D).
    Args:
        batch_size (int): The batch size.
        latent_vector_size (int): The latent vector size.
        dimension (int): The dimension of the images in a given problem.
    can be 2 for 2D or 3 for 3D.
        device (str): The device to perform computations on.
    Returns:
        latent_vector (torch.Tensor): The latent vector.
    """
    assert dimension in [2, 3], "Dimension should be 2 (2D) or 3 (3D)"
    latent_vector = torch.randn(
        (batch_size, latent_vector_size, 1, 1), device=device
    )
    if dimension == 3:
        latent_vector = latent_vector.unsqueeze(-1)
    return latent_vector


def get_fixed_latent_vector(
    batch_size: int,
    latent_vector_size: int,
    dimension: int,
    device: str,
    seed: int,
) -> torch.Tensor:
    """
    Function to get the fixed latent vector for inference or validation.
    It always starts with the seed given by user and then re-sets the
    previous RNG state.
    Args:
        batch_size (int): The batch size.
        latent_vector_size (int): The latent vector size.
        dimension (int): The dimension of the images in a given problem. Can be `2` for 2D or `3` for 3D.
        device (str): The device to perform computations on.
        seed (int): The seed to use for reproducibility.
    Returns:
        latent_vector (torch.Tensor): The fixed latent vector.
    """
    assert dimension in [2, 3], "Dimension should be 2 (2D) or 3 (3D)"
    current_rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    latent_vector = torch.randn(
        (batch_size, latent_vector_size, 1, 1),
        device=device,
    )
    if dimension == 3:
        latent_vector = latent_vector.unsqueeze(-1)
    torch.set_rng_state(current_rng_state)
    return latent_vector


def create_pytorch_objects_gan(
    parameters, train_csv=None, val_csv=None, device="cpu"
):
    """
    This function creates all the PyTorch objects needed for training.

    Args:
        parameters (dict): The parameters dictionary.
        train_csv (str): The path to the training CSV file.
        val_csv (str): The path to the validation CSV file.
        device (str): The device to perform computations on.

    Returns:
        model (torch.nn.Module): The model to use for training.
        optimizer_gen (Optimizer): The optimizer to use for training generator.
        optimizer_disc (Optimizer): The optimizer to use for training discriminator.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        scheduler_gen (object): The scheduler to use for training generator.
        scheduler_disc (object): The scheduler to use for training discriminator.
        parameters (dict): The updated parameters dictionary.
    """

    # initialize train and val loaders
    train_loader, val_loader = None, None
    headers_to_populate_train, headers_to_populate_val = None, None

    if train_csv is not None:
        # populate the data frames
        (
            parameters["training_data"],
            headers_to_populate_train,
        ) = parseTrainingCSV(train_csv, train=True)
        parameters = populate_header_in_parameters(
            parameters, headers_to_populate_train
        )
        # get the train loader
        train_loader = get_train_loader(parameters)
        parameters["training_samples_size"] = len(train_loader)
        # get the hash of the training data for reproducibility
        parameters["training_data_hash"] = hash_pandas_object(
            parameters["training_data"]
        ).sum()

    if val_csv is not None:
        (
            parameters["validation_data"],
            headers_to_populate_val,
        ) = parseTrainingCSV(val_csv, train=False)
        if headers_to_populate_train is None:
            parameters = populate_header_in_parameters(
                parameters, headers_to_populate_val
            )
        # get the validation loader
        val_loader = get_validation_loader(parameters)

    # get the model
    model = get_model(parameters)

    parameters["model_parameters_gen"] = model.generator.parameters()
    parameters["model_parameters_disc"] = model.discriminator.parameters()

    optimizer_gen, optimizer_disc = get_optimizers_gan(parameters)
    parameters["optimizer_gen_object"] = optimizer_gen
    parameters["optimizer_disc_object"] = optimizer_disc
    (
        model,
        parameters["model"]["amp"],
        parameters["device"],
        parameters["device_id"],
    ) = send_model_to_device(
        model,
        amp=parameters["model"]["amp"],
        device=device,
        optimizer_1=optimizer_gen,
        optimizer_2=optimizer_disc,
    )
    if train_csv is not None:
        if not ("step_size" in parameters["scheduler_g"]):
            parameters["scheduler_g"]["step_size"] = (
                parameters["training_samples_size"]
                / parameters["learning_rate_g"]
            )
        if not ("step_size" in parameters["scheduler_d"]):
            parameters["scheduler_g"]["step_size"] = (
                parameters["training_samples_size"]
                / parameters["learning_rate_d"]
            )

        scheduler_gen, scheduler_disc = get_scheduler_gan(parameters)

        # Calculate the weights here
        (
            parameters["weights"],
            parameters["class_weights"],
        ) = get_class_imbalance_weights(
            parameters["training_data"], parameters
        )

        print("Class weights  : ", parameters["class_weights"])
        print("Penalty weights: ", parameters["weights"])

    else:
        scheduler_gen, scheduler_disc = None, None

    # these keys contain generators, and are not needed beyond this point in params
    generator_keys_to_remove = [
        "optimizer_gen_object",
        "optimizer_disc_object",
        "model_parameters_gen",
        "model_parameters_disc",
    ]
    for key in generator_keys_to_remove:
        parameters.pop(key, None)
    return (
        model,
        optimizer_gen,
        optimizer_disc,
        train_loader,
        val_loader,
        scheduler_gen,
        scheduler_disc,
        parameters,
    )
