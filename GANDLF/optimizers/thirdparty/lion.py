from torch.optim.optimizer import Optimizer
from lion_pytorch import Lion


def lion_wrapper(parameters: dict) -> Optimizer:
    """
    Creates an instance of the Lion optimizer from the `lion_pytorch` package using the input parameters.

    Args:
        parameters (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        Optimizer: An instance of the Lion optimizer.
    """
    return Lion(
        parameters["model_parameters"],
        lr=parameters.get("learning_rate", 1e-4),
        betas=parameters["optimizer"].get("betas", (0.9, 0.999)),
        weight_decay=parameters["optimizer"].get("weight_decay", 0.0),
        decoupled_weight_decay=parameters["optimizer"].get(
            "decoupled_weight_decay", False
        ),
        use_triton=False,  # as of 20241120, triton is not generally available for all platforms
    )
