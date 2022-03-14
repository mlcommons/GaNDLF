from .unet import unet, resunet
from .light_unet import light_unet, light_resunet
from .deep_unet import deep_unet, deep_resunet
from .uinc import uinc
from .fcn import fcn
from .vgg import vgg11, vgg13, vgg16, vgg19
from .densenet import densenet121, densenet169, densenet201, densenet264
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .efficientnet import (
    efficientnetB0,
    efficientnetB1,
    efficientnetB2,
    efficientnetB3,
    efficientnetB4,
    efficientnetB5,
    efficientnetB6,
    efficientnetB7,
)
from .sdnet import SDNet
from .MSDNet import MSDNet
from .brain_age import brainage

# defining dict for models - key is the string and the value is the transform object
global_models_dict = {
    "unet": unet,
    "resunet": resunet,
    "residualunet": resunet,
    "deepunet": deep_unet,
    "lightunet": light_unet,
    "deep_unet": deep_unet,
    "light_unet": light_unet,
    "deepresunet": deep_resunet,
    "lightresunet": light_resunet,
    "deep_resunet": deep_resunet,
    "light_resunet": light_resunet,
    "fcn": fcn,
    "uinc": uinc,
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
    "brain_age": brainage,
    "sdnet": SDNet,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "efficientnetb0": efficientnetB0,
    "efficientnetb1": efficientnetB1,
    "efficientnetb2": efficientnetB2,
    "efficientnetb3": efficientnetB3,
    "efficientnetb4": efficientnetB4,
    "efficientnetb5": efficientnetB5,
    "efficientnetb6": efficientnetB6,
    "efficientnetb7": efficientnetB7,
}
