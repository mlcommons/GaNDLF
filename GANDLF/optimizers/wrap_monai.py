import monai
from monai.optimizers import Novograd


def novograd_wrapper(parameters: dict) -> monai.optimizers.Novograd:
    return Novograd(
        parameters["model_parameters"],
        lr=parameters.get("learning_rate", 1e-3),
        betas=parameters["optimizer"].get("betas", (0.9, 0.999)),
        eps=parameters["optimizer"].get("eps", 1e-8),
        weight_decay=parameters["optimizer"].get("weight_decay", 3e-05),
        amsgrad=parameters["optimizer"].get("amsgrad", False),
    )
