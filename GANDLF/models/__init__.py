from .unet import unet, resunet
from .uinc import uinc
from .fcn import fcn
from .vgg import vgg11, vgg13, vgg16, vgg19
from .sdnet import SDNet

global_models_dict = {
    "unet": unet,
    "resunet": resunet,
    "residualunet": unet,
    "fcn": fcn,
    "uinc": uinc,
    "sdnet": SDNet,
    "vgg": vgg19,
    "vgg11": vgg11,
    "vgg13": vgg13,
    "vgg16": vgg16,
    "vgg19": vgg19,
}

def get_model(parameters):
    model = global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    # special case for unet variants
    if ("residual" in parameters["model"]["architecture"]) or ("resunet" in parameters["model"]["architecture"]):
        model = global_models_dict["unet"](parameters=parameters, residualConnections=True)

    return model