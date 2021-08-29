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
