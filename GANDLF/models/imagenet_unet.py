# -*- coding: utf-8 -*-
# adapted from https://github.com/qubvel/segmentation_models.pytorch
import segmentation_models_pytorch as smp
from .modelBase import ModelBase


class ImageNet_UNet(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        super(ImageNet_UNet, self).__init__(parameters)

        # define a default encoder
        # all encoders: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/encoders/__init__.py
        encoder_name = parameters["model"].get("encoder_name", "resnet34")
        decoder_attention_type = parameters["model"].get("decoder_attention_type", None)
        decoder_use_batchnorm = False
        if parameters["model"]["norm_type"] == "batch":
            decoder_use_batchnorm = True
        decoder_use_batchnorm = parameters["model"].get("decoder_use_batchnorm", False)

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=self.n_channels,
            classes=self.n_classes,
            activation=parameters["model"]["final_layer"],
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_attention_type=decoder_attention_type,
        )

    def forward(self, x):
        return self.model(x)


def imagenet_unet_wrapper(parameters):
    return ImageNet_UNet(parameters)
