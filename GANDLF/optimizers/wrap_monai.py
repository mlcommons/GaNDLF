from monai.optimizers import (
    novograd,
    WarmupCosineSchedule as WCS
)

def novograd(parameters):
    return novograd(
        parameters["model_parameters"],
        lr=parameters.get("learning_rate"),
        betas=parameters["optimizer"].get("betas", (0.9, 0.999)),
        eps=parameters["optimizer"].get("eps", 1e-8),
        weight_decay=parameters["optimizer"].get("weight_decay", 3e-05),
        amsgrad=parameters["optimizer"].get("amsgrad", False),
    )

def WCS(parameters):
    return WCS(
        
    )
