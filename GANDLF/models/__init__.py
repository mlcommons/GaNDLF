# Import the different model architectures and functions
from .unet import unet, resunet
from .light_unet import light_unet, light_resunet
from .unet_multilayer import unet_multilayer, resunet_multilayer
from .light_unet_multilayer import light_unet_multilayer, light_resunet_multilayer
from .deep_unet import deep_unet, deep_resunet
from .uinc import uinc
from .fcn import fcn
from .vgg import vgg11, vgg13, vgg16, vgg19
from .densenet import densenet121, densenet169, densenet201, densenet264
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnet200
from .dynunet_wrapper import dynunet_wrapper
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
from .imagenet_vgg import (
    imagenet_vgg11,
    imagenet_vgg11_bn,
    imagenet_vgg13,
    imagenet_vgg13_bn,
    imagenet_vgg16,
    imagenet_vgg16_bn,
    imagenet_vgg19,
    imagenet_vgg19_bn,
)
from .imagenet_unet import imagenet_unet_wrapper
from .sdnet import SDNet
from .MSDNet import MSDNet
from .brain_age import brainage
from .unetr import unetr
from .transunet import transunet
from .modelBase import ModelBase

# Define a dictionary of model architectures and corresponding functions
global_models_dict = {
    # Types of unet
    "unet": unet,
    "unet_multilayer": unet_multilayer,
    "resunet": resunet,
    "resunet_multilayer": resunet_multilayer,
    "residualunet": resunet,  # Alias for "resunet"
    "residualunet_multilayer": resunet_multilayer,  # Alias for "resunet_multilayer"
    "deepunet": deep_unet,
    "lightunet": light_unet,
    "lightunet_multilayer": light_unet_multilayer,
    "deep_unet": deep_unet,  # Alias for "deepunet"
    "light_unet": light_unet,  # Alias for "lightunet"
    "light_unet_multilayer": light_unet_multilayer,  # Alias for "lightunet_multilayer"
    "deepresunet": deep_resunet,
    "lightresunet": light_resunet,
    "lightresunet_multilayer": light_resunet_multilayer,
    "deep_resunet": deep_resunet,  # Alias for "deepresunet"
    "light_resunet": light_resunet,  # Alias for "lightresunet"
    "light_resunet_multilayer": light_resunet_multilayer,  # Alias for "lightresunet_multilayer"
    "unetr": unetr,
    "transunet": transunet,
    "uinc": uinc,
    # UNet models with imagenet support from segmentation_models.pytorch
    "imagenet_unet": imagenet_unet_wrapper,
    # Additional segmentation model
    "fcn": fcn,
    # VGG models
    "vgg": vgg19,
    "vgg11": vgg11,
    "vgg13": vgg13,
    "vgg16": vgg16,
    "vgg19": vgg19,
    # VGG models with imagenet support
    "imagenet_vgg11": imagenet_vgg11,
    "imagenet_vgg11_bn": imagenet_vgg11_bn,
    "imagenet_vgg13": imagenet_vgg13,
    "imagenet_vgg13_bn": imagenet_vgg13_bn,
    "imagenet_vgg16": imagenet_vgg16,
    "imagenet_vgg16_bn": imagenet_vgg16_bn,
    "imagenet_vgg19": imagenet_vgg19,
    "imagenet_vgg19_bn": imagenet_vgg19_bn,
    # DenseNet models
    "densenet": densenet264,
    "densenet121": densenet121,
    "densenet169": densenet169,
    "densenet201": densenet201,
    "densenet264": densenet264,
    # ResNet models
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnet200": resnet200,
    # EfficientNet models
    "efficientnetb0": efficientnetB0,
    "efficientnetb1": efficientnetB1,
    "efficientnetb2": efficientnetB2,
    "efficientnetb3": efficientnetB3,
    "efficientnetb4": efficientnetB4,
    "efficientnetb5": efficientnetB5,
    "efficientnetb6": efficientnetB6,
    "efficientnetb7": efficientnetB7,
    "dynunet": dynunet_wrapper,
    # Custom models
    "msdnet": MSDNet,
    "brain_age": brainage,
    "sdnet": SDNet,
}


def get_model(params: dict) -> ModelBase:
    """
    Function to get the model definition.

    Args:
        params (dict): The parameters' dictionary.

    Returns:
        model (ModelBase): The model definition.
    """
    chosen_model = params["model"]["architecture"].lower()
    assert (
        chosen_model in global_models_dict
    ), f"Could not find the requested model '{params['model']['architecture']}'"
    return global_models_dict[chosen_model](parameters=params)
