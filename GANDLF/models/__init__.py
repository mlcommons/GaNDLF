from .unet import unet, resunet
from .uinc import uinc
from .fcn import fcn
from .vgg import vgg11, vgg13, vgg16, vgg19
from .densenet import densenet121, densenet169, densenet201, densenet264
from .sdnet import SDNet
from .MSDNet import MSDNet
from .brain_age import get_model

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
    "densenet": densenet264,
    "densenet121": densenet121,
    "densenet169": densenet169,
    "densenet201": densenet201,
    "densenet264": densenet264,
    "msdnet": MSDNet,
    "brain_age": get_model,
}
