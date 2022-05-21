# -*- coding: utf-8 -*-
# adapted from https://github.com/qubvel/segmentation_models.pytorch
import segmentation_models_pytorch as smp
from acsconv.converters import ACSConverter, Conv3dConverter

from .modelBase import ModelBase


class ImageNet_UNet(ModelBase):
    def __init__(
        self,
        parameters,
    ) -> None:
        super(ImageNet_UNet, self).__init__(parameters)

        decoder_use_batchnorm = False
        if parameters["model"]["norm_type"] == "batch":
            decoder_use_batchnorm = True
        decoder_use_batchnorm = parameters["model"].get(
            "decoder_use_batchnorm", decoder_use_batchnorm
        )

        self.model = smp.Unet(
            encoder_name=parameters["model"].get("encoder_name", "resnet152"),
            encoder_weights="imagenet",
            in_channels=self.n_channels,
            classes=self.n_classes,
            activation=parameters["model"]["final_layer"],
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_attention_type=parameters["model"].get(
                "decoder_attention_type", None
            ),
            encoder_depth=parameters["model"].get("encoder_depth", 5),
            decoder_channels=parameters["model"].get(
                "decoder_channels", (256, 128, 64, 32, 16)
            ),
        )

        if self.n_dimensions == 3:
            self.model = ACSConverter(self.model).model

    def forward(self, x):
        return self.model(x)


def imagenet_unet_wrapper(parameters):
    return ImageNet_UNet(parameters)
