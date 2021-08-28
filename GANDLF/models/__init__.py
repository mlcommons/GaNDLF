from .unet import unet

global_models_dict = {
    "unet": unet,
    "resunet": unet,
    "residualunet": unet,
}

def get_model(parameters):
    model = global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    # special case for unet variants
    if ("residual" in parameters["model"]["architecture"]) or ("resunet" in parameters["model"]["architecture"]):
        model = global_models_dict["unet"](parameters=parameters, residualConnections=True)

    return model