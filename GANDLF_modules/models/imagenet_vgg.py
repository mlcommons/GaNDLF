# -*- coding: utf-8 -*-
"""
Modified from https://github.com/pytorch/vision.git
"""

import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .modelBase import ModelBase


def create_torchvision_model(modelname, pretrained=True, num_classes=2, dimensions=2):
    """
    Create a torchvision model with the given parameters.

    Args:
        modelname (str): The model to create.
        pretrained (bool, optional): If pretrained model is to be returned. Defaults to True.
        num_classes (int, optional): The number of output classes. Defaults to 2.
        dimensions (int, optional): The dimensionality of computations. Defaults to 2.

    Returns:
        model (torchvision.models.model): The created model after taking output classes into account.
    """

    assert dimensions == 2, "ImageNet_VGG only supports 2D images"

    if modelname == "vgg11":
        model = torchvision.models.vgg11(
            pretrained=pretrained,
        )
    if modelname == "vgg11_bn":
        model = torchvision.models.vgg11_bn(
            pretrained=pretrained,
        )
    if modelname == "vgg13":
        model = torchvision.models.vgg13(
            pretrained=pretrained,
        )
    if modelname == "vgg13_bn":
        model = torchvision.models.vgg13_bn(
            pretrained=pretrained,
        )
    if modelname == "vgg16":
        model = torchvision.models.vgg16(
            pretrained=pretrained,
        )
    if modelname == "vgg16_bn":
        model = torchvision.models.vgg16_bn(
            pretrained=pretrained,
        )
    if modelname == "vgg19":
        model = torchvision.models.vgg19(
            pretrained=pretrained,
        )
    if modelname == "vgg19_bn":
        model = torchvision.models.vgg19_bn(
            pretrained=pretrained,
        )
    model.classifier[6] = nn.Linear(
        in_features=model.classifier[3].out_features, out_features=num_classes
    )
    return model


def apply_activation_function(activation_function, input_tensor):
    """
    Applies the given activation function to the input tensor.

    Args:
        activation_function (callable or None): Activation function to apply.
        input_tensor (torch.Tensor): Input tensor to apply the activation function to.

    Returns:
        torch.Tensor: Output tensor after applying the activation function.
    """
    out = input_tensor
    if not activation_function is None:
        if activation_function == F.softmax:
            out = activation_function(out, dim=1)
        else:
            out = activation_function(out)
    return out


class imagenet_vgg11(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        """
        Implements the VGG-11 model for ImageNet classification.

        Args:
            parameters (dict): Dictionary of parameters to initialize the model.

        Attributes:
            model (torch.nn.Module): The VGG-11 model.
        """
        super(imagenet_vgg11, self).__init__(parameters)

        pretrained = parameters["model"].get("pretrained", True)
        self.model = create_torchvision_model(
            "vgg11",
            pretrained=pretrained,
            num_classes=self.n_classes,
            dimensions=self.n_dimensions,
        )

    def forward(self, x):
        """
        Forward pass of the VGG-11 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the final activation function.
        """
        return apply_activation_function(self.final_convolution_layer, self.model(x))


class imagenet_vgg11_bn(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        """
        VGG11 with batch normalization architecture for image classification on ImageNet dataset.

        Args:
            parameters (dict): Dictionary containing model parameters.

        Attributes:
            model (torch.nn.Module): VGG11 with batch normalization model.
        """
        super(imagenet_vgg11_bn, self).__init__(parameters)

        pretrained = parameters["model"].get("pretrained", True)
        self.model = create_torchvision_model(
            "vgg11_bn",
            pretrained=pretrained,
            num_classes=self.n_classes,
            dimensions=self.n_dimensions,
        )

    def forward(self, x):
        """
        Forward pass method.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return apply_activation_function(self.final_convolution_layer, self.model(x))


class imagenet_vgg13(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        """
        Implements the VGG-13 model for ImageNet classification.

        Args:
            parameters (dict): Dictionary of parameters to initialize the model.

        Attributes:
            model (torch.nn.Module): The VGG-13 model.
        """
        super(imagenet_vgg13, self).__init__(parameters)

        pretrained = parameters["model"].get("pretrained", True)
        self.model = create_torchvision_model(
            "vgg13",
            pretrained=pretrained,
            num_classes=self.n_classes,
            dimensions=self.n_dimensions,
        )

    def forward(self, x):
        """
        Forward pass of the VGG-13 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the final activation function.
        """
        return apply_activation_function(self.final_convolution_layer, self.model(x))


class imagenet_vgg13_bn(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        """
        VGG13 with batch normalization architecture for image classification on ImageNet dataset.

        Args:
            parameters (dict): Dictionary containing model parameters.

        Attributes:
            model (torch.nn.Module): VGG13 with batch normalization model.
        """
        super(imagenet_vgg13_bn, self).__init__(parameters)

        pretrained = parameters["model"].get("pretrained", True)
        self.model = create_torchvision_model(
            "vgg13_bn",
            pretrained=pretrained,
            num_classes=self.n_classes,
            dimensions=self.n_dimensions,
        )

    def forward(self, x):
        """
        Forward pass method.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return apply_activation_function(self.final_convolution_layer, self.model(x))


class imagenet_vgg16(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        """
        Implements the VGG-16 model for ImageNet classification.

        Args:
            parameters (dict): Dictionary of parameters to initialize the model.

        Attributes:
            model (torch.nn.Module): The VGG-16 model.
        """
        super(imagenet_vgg16, self).__init__(parameters)

        pretrained = parameters["model"].get("pretrained", True)
        self.model = create_torchvision_model(
            "vgg16",
            pretrained=pretrained,
            num_classes=self.n_classes,
            dimensions=self.n_dimensions,
        )

    def forward(self, x):
        """
        Forward pass of the VGG-16 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the final activation function.
        """
        return apply_activation_function(self.final_convolution_layer, self.model(x))


class imagenet_vgg16_bn(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        """
        VGG16 with batch normalization architecture for image classification on ImageNet dataset.

        Args:
            parameters (dict): Dictionary containing model parameters.

        Attributes:
            model (torch.nn.Module): VGG16 with batch normalization model.
        """
        super(imagenet_vgg16_bn, self).__init__(parameters)

        pretrained = parameters["model"].get("pretrained", True)
        self.model = create_torchvision_model(
            "vgg16_bn",
            pretrained=pretrained,
            num_classes=self.n_classes,
            dimensions=self.n_dimensions,
        )

    def forward(self, x):
        """
        Forward pass method.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return apply_activation_function(self.final_convolution_layer, self.model(x))


class imagenet_vgg19(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        """
        Implements the VGG-19 model for ImageNet classification.

        Args:
            parameters (dict): Dictionary of parameters to initialize the model.

        Attributes:
            model (torch.nn.Module): The VGG-19 model.
        """
        super(imagenet_vgg19, self).__init__(parameters)

        pretrained = parameters["model"].get("pretrained", True)
        self.model = create_torchvision_model(
            "vgg19",
            pretrained=pretrained,
            num_classes=self.n_classes,
            dimensions=self.n_dimensions,
        )

    def forward(self, x):
        """
        Forward pass of the VGG-19 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the final activation function.
        """
        return apply_activation_function(self.final_convolution_layer, self.model(x))


class imagenet_vgg19_bn(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        """
        VGG19 with batch normalization architecture for image classification on ImageNet dataset.

        Args:
            parameters (dict): Dictionary containing model parameters.

        Attributes:
            model (torch.nn.Module): VGG19 with batch normalization model.
        """
        super(imagenet_vgg19_bn, self).__init__(parameters)

        pretrained = parameters["model"].get("pretrained", True)
        self.model = create_torchvision_model(
            "vgg19_bn",
            pretrained=pretrained,
            num_classes=self.n_classes,
            dimensions=self.n_dimensions,
        )

    def forward(self, x):
        """
        Forward pass method.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return apply_activation_function(self.final_convolution_layer, self.model(x))
