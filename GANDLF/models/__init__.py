from GANDLF.losses.gan import LSGAN_loss
from GANDLF.models.discriminators.dcgan import DCGANDiscriminator
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

from .discriminators.vanillaGAN import vanillaGANDiscriminator
from .discriminators.dcgan import DCGANDiscriminator
from .discriminators.lsgan import LSGANDiscriminator

from .generators.vanillaGAN import vanillaGANGenerator
from .generators.dcgan import DCGANGenerator
from .generators.lsgan import LSGANGenerator

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
}

global_discriminators_dict = {
    "gan": vanillaGANDiscriminator,
    "dcgan": DCGANDiscriminator,
    "lsgan": LSGANDiscriminator
}

global_generators_dict = {
    "gan": vanillaGANGenerator,
    "dcgan": DCGANGenerator,
    "lsgan": LSGANGenerator
}