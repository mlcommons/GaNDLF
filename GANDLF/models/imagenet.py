# -*- coding: utf-8 -*-
"""
Modified from https://github.com/pytorch/vision.git
"""

import torchvision
import torch.nn as nn

from .modelBase import ModelBase


def create_torchvision_model(modelname, pretrained=True, num_classes=2):
    if modelname == "vgg16":
        model = torchvision.models.vgg16(pretrained=pretrained)
    if modelname == "vgg16_bn":
        model = torchvision.models.vgg16_bn(pretrained=pretrained)
    if modelname == "vgg19":
        model = torchvision.models.vgg19(pretrained=pretrained)
    if modelname == "vgg19_bn":
        model = torchvision.models.vgg19_bn(pretrained=pretrained)
    prev_out_features = model.classifier[3].out_features
    model.classifier[6] = nn.Linear(
        in_features=prev_out_features, out_features=num_classes
    )
    return model


class imagenet_vgg16(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        super().__init__()
        self.model = create_torchvision_model(
            "vgg16", pretrained=True, num_classes=self.n_classes
        )

    def forward(self, x):
        return self.model(x)


class imagenet_vgg19(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        super().__init__()
        self.model = create_torchvision_model(
            "vgg19", pretrained=True, num_classes=self.n_classes
        )

    def forward(self, x):
        return self.model(x)


class imagenet_vgg16_bn(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        super().__init__()
        self.model = create_torchvision_model(
            "vgg16_bn", pretrained=True, num_classes=self.n_classes
        )

    def forward(self, x):
        return self.model(x)


class imagenet_vgg19_bn(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        super().__init__()
        self.model = create_torchvision_model(
            "vgg19_bn", pretrained=True, num_classes=self.n_classes
        )

    def forward(self, x):
        return self.model(x)
