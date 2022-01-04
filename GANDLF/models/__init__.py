from .unet import unet, resunet
from .light_unet import light_unet, light_resunet
from .deep_unet import deep_unet, deep_resunet
from .uinc import uinc
from .fcn import fcn
from .vgg import vgg11, vgg13, vgg16, vgg19
from .densenet import densenet121, densenet169, densenet201, densenet264
from .sdnet import SDNet
from .MSDNet import MSDNet
from .brain_age import brainage
from .pix2pix import pix2pix
from .pix2pixHD import pix2pixHD
from .cycleGAN import cycleGAN
from .dcgan import dcgan

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
}

global_gan_models_dict = {
    "sdnet": SDNet,
    "pix2pix": pix2pix,
    "pix2pixHD": pix2pixHD,
    "cycleGAN": cycleGAN,
    "dcgan": dcgan,
}
