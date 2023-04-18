# -*- coding: utf-8 -*-
# adapted from https://github.com/qubvel/segmentation_models.pytorch
from typing import Optional, Union, List
import torch
import torch.nn as nn

from segmentation_models_pytorch.base import (
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import initialization as init

from .modelBase import ModelBase


class SegmentationModel(torch.nn.Module):
    """
    This class defines a segmentation model. It takes an input tensor and sequentially passes it through the model's
    encoder, decoder and heads.

    Methods:
        initialize:
            Initializes the model's classification or segmentation head

        forward:
            Sequentially passes the input tensor through the model's encoder, decoder, and heads and returns either the
            segmentation masks or the class labels.
    """

    def initialize(self):
        """
        This function initializes the model's classification or segmentation head.
        """
        if self.classification_head is None:
            init.initialize_decoder(self.decoder)
            init.initialize_head(self.segmentation_head)
        else:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """
        This function sequentially passes the input tensor through the model's encoder, decoder, and heads and returns
        either the segmentation masks or the class labels.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Segmentation masks or class labels
        """
        features = self.encoder(x)

        if self.classification_head is None:
            decoder_output = self.decoder(*features)
            masks = self.segmentation_head(decoder_output)
            return masks
        else:
            labels = self.classification_head(features[-1])
            return labels

    ## commented out because we are not using this interface
    # def predict(self, x):
    #     """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

    #     Args:
    #         x: 4D torch tensor with shape (batch_size, channels, height, width)

    #     Return:
    #         prediction: 4D torch tensor with shape (batch_size, classes, height, width)

    #     """
    #     if self.training:
    #         self.eval()

    #     with torch.no_grad():
    #         x = self.forward(x)

    #     return x


class Unet(SegmentationModel):
    """
    This has been adapted from its original implementation in
    https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/decoders/unet/model.py
    Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name (str): Name of the classification model to be used as an encoder (a.k.a backbone) to extract features of different spatial resolution.
        encoder_depth (int): A number of stages used in encoder in range [3, 5]. Each stage generates features two times smaller in spatial dimensions than the previous one. Default is 5.
        encoder_weights (str or None): One of "None" (random initialization), "imagenet" (pre-training on ImageNet) and other pretrained weights (see table with available weights for each encoder_name).
        decoder_channels (list of ints): List of integers specifying in_channels parameter for convolutions used in decoder. Length of the list should be the same as encoder_depth.
        decoder_use_batchnorm (bool or str): If True, BatchNorm2d layer between Conv2D and Activation layers is used. If "inplace", InplaceABN will be used, allowing to decrease memory consumption. Available options are True, False, "inplace".
        decoder_attention_type (str or None): Attention module used in the decoder of the model. Available options are None and "scse". SCSE paper - https://arxiv.org/abs/1808.08127.
        in_channels (int): A number of input channels for the model. Default is 3 (RGB images).
        classes (int): A number of classes for output mask (or you can think as a number of channels of output mask).
        activation (str, callable, or None): An activation function to apply after the final convolution layer. Available options are "sigmoid", "softmax", "logsoftmax", "tanh", "identity", callable, and None. Default is None.
        aux_params (dict or None): Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is built on top of the encoder if aux_params is not None (default). Supported params:
            - classes (int): A number of classes.
            - pooling (str): One of "max", "avg". Default is "avg".
            - dropout (float): Dropout factor in [0, 1).
            - activation (str): An activation function to apply "sigmoid"/"softmax" (could be None to return logits).

    Returns:
        model (torch.nn.Module): A UNet model for image semantic segmentation.

    Reference:
    Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
    arXiv preprint arXiv:1505.04597.

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        def _get_fixed_encoder_channels(name, in_channels):
            if "mit_" in name:
                # MixVision Transformers only support 3-channel inputs
                return 3
            else:
                return in_channels

        self.encoder = get_encoder(
            encoder_name,
            in_channels=_get_fixed_encoder_channels(encoder_name, in_channels),
            depth=encoder_depth,
            weights=encoder_weights,
        )
        out_channels = self.encoder.out_channels
        if "mit_" in encoder_name:
            # MixVision Transformers only support 3-channel inputs
            print(
                "WARNING: MixVision Transformers only support 3 channels, adding an extra Conv layer for compatibility."
            )
            self.pre_encoder = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1)
            modules = []
            modules.append(self.pre_encoder)
            modules.append(self.encoder)
            self.encoder = nn.Sequential(*modules)

        if aux_params is None:
            self.decoder = UnetDecoder(
                encoder_channels=out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=decoder_attention_type,
            )
            self.segmentation_head = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=classes,
                activation=activation,
                kernel_size=3,
            )
            self.classification_head = None
        else:
            self.classification_head = ClassificationHead(
                in_channels=out_channels[-1], **aux_params
            )
            self.segmentation_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()


class ImageNet_UNet(ModelBase):
    """
    Wrapper class to parse arguments from configuration and construct appropriate model object.

    Args:
        ModelBase (nn.Module): The base model class.
    """

    def __init__(
        self,
        parameters,
    ) -> None:
        super(ImageNet_UNet, self).__init__(parameters)

        # https://github.com/qubvel/segmentation_models.pytorch/issues/745
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

        decoder_use_batchnorm = False
        if parameters["model"]["norm_type"] == "batch":
            decoder_use_batchnorm = True
        decoder_use_batchnorm = parameters["model"].get(
            "decoder_use_batchnorm", decoder_use_batchnorm
        )

        default_encoder_name = parameters["model"].get("encoder_name", "resnet34")
        # MixVision Transformers only support 2D inputs with 3 channels (the channel dimension is checked in the encoder)
        if ("mit_" in default_encoder_name) or ("timm-" in default_encoder_name):
            assert (
                self.n_dimensions == 2
            ), "MixVision Transformers and TIMM models only support 2D inputs"

        encoder_depth = parameters["model"].get("depth", 5)
        encoder_depth = parameters["model"].get("encoder_depth", encoder_depth)
        parameters["model"]["pretrained"] = parameters["model"].get("pretrained", True)
        if parameters["model"]["pretrained"]:
            parameters["model"]["encoder_weights"] = "imagenet"
            # special case for some encoders: https://github.com/search?q=repo%3Aqubvel%2Fsegmentation_models.pytorch%20imagenet%2B5k&type=code
            if default_encoder_name in ["dpn68b", "dpn92", "dpn107"]:
                parameters["model"]["encoder_weights"] = "imagenet+5k"
        else:
            parameters["model"]["encoder_weights"] = None

        if parameters["problem_type"] != "segmentation":
            classifier_head_parameters = {}
            classifier_head_parameters["classes"] = self.n_classes
            classifier_head_parameters["activation"] = parameters["model"][
                "final_layer"
            ]
            if classifier_head_parameters["activation"] == "None":
                classifier_head_parameters["activation"] = None
            classifier_head_parameters["dropout"] = parameters["model"].get(
                "dropout", 0.2
            )
            classifier_head_parameters["pooling"] = parameters["model"].get(
                "pooling", "avg"
            )
        else:
            classifier_head_parameters = None

        self.model = Unet(
            encoder_name=default_encoder_name,
            encoder_weights=parameters["model"]["encoder_weights"],
            in_channels=self.n_channels,
            classes=self.n_classes,
            activation=parameters["model"]["final_layer"],
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_attention_type=parameters["model"].get(
                "decoder_attention_type", None
            ),
            encoder_depth=encoder_depth,
            decoder_channels=parameters["model"].get(
                "decoder_channels", (256, 128, 64, 32, 16)
            ),
            aux_params=classifier_head_parameters,
        )

        if self.n_dimensions == 3:
            self.model = self.converter(self.model).model

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


def imagenet_unet_wrapper(parameters):
    return ImageNet_UNet(parameters)
