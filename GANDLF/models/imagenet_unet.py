# -*- coding: utf-8 -*-
# adapted from https://github.com/qubvel/segmentation_models.pytorch
import segmentation_models_pytorch as smp
from .modelBase import ModelBase


class imagenet_unet(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        super(imagenet_unet, self).__init__(parameters)

        # define a default encoder
        # all encoders: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/encoders/__init__.py
        encoder_name = parameters.get("encoder_name", "resnet34")
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=self.n_channels,
            classes=self.n_classes,
        )
