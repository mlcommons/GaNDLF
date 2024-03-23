import typing, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from GANDLF.models.seg_modules.add_conv_block import add_conv_block
from GANDLF.models.seg_modules.add_downsample_conv_block import (
    add_downsample_conv_block,
)
from GANDLF.models.unet import unet
from .modelBase import ModelBase
from copy import deepcopy


class Decoder(ModelBase):
    def __init__(self, parameters, anatomy_factors, num_layers=5):
        """
        Decoder module for SDNet.

        Args:
            parameters (dict): A dictionary containing model parameters.
            anatomy_factors (int): The number of anatomical factors to be considered.
            num_layers (int, optional): The number of layers in the Decoder. Defaults to 5.

        Attributes:
            num_layers (int): The number of layers in the Decoder.
            layer_list (list): List of layers in the Decoder module.
            conv (nn.Conv2d): Convolutional layer to generate the final output.
            layers (nn.ModuleList): List of layers in the Decoder module.
        """
        super(Decoder, self).__init__(parameters)
        self.num_layers = num_layers
        self.layer_list = add_conv_block(
            self.Conv,
            self.BatchNorm,
            in_channels=anatomy_factors,
            out_channels=self.base_filters,
        )
        for _ in range(self.num_layers - 2):
            self.layer_list += add_conv_block(
                self.Conv,
                self.BatchNorm,
                in_channels=self.base_filters,
                out_channels=self.base_filters,
            )
        self.conv = self.Conv(self.base_filters, 1, 3, 1, 1)
        # Add layers to Module List
        self.layers = nn.ModuleList(self.layer_list)
        self.apply(self.weight_init)

        nn.init.xavier_normal_(self.conv.weight.data)
        self.conv.bias.data.zero_()

    @staticmethod
    def CalcVectorMeanStd(feat, eps=1e-5):
        """
        Calculate the mean and standard deviation of the input feature vector.

        Args:
            feat (torch.Tensor): Input feature vector.
            eps (float, optional): Small value added to the variance to avoid divide-by-zero. Defaults to 1e-5.

        Returns:
            tuple: Tuple containing the mean and standard deviation of the input feature vector.
        """

        # eps is a small value added to the variance to avoid divide-by-zero.
        feat_var = feat.var(dim=1) + eps
        feat_std = feat_var.sqrt()
        feat_mean = feat.mean(dim=1)
        return feat_mean, feat_std

    @staticmethod
    def CalcTensorMeanStd(feat, eps=1e-5):
        """
        Calculate the mean and standard deviation of the input feature tensor.

        Args:
            feat (torch.Tensor): Input feature tensor.
            eps (float, optional): Small value added to the variance to avoid divide-by-zero. Defaults to 1e-5.

        Returns:
            tuple: Tuple containing the mean and standard deviation of the input feature tensor.
        """
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert len(size) == 4
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    @staticmethod
    def weight_init(model):
        """
        Initialize weights for the given model using He (Kaiming) initialization.

        Args:
            model (nn.Module): Model for which the weights will be initialized.
        """
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    @staticmethod
    def AdaIN(content_feat, style_feat):
        """
        Adaptive Instance Normalization (AdaIN) operation.

        Args:
            content_feat (torch.Tensor): Content feature tensor.
            style_feat (torch.Tensor): Style feature tensor.

        Returns:
            torch.Tensor: Tensor after applying AdaIN operation.
        """
        size = content_feat.size()
        style_mean, style_std = Decoder.CalcVectorMeanStd(style_feat)
        content_mean, content_std = Decoder.CalcTensorMeanStd(content_feat)
        normalized_feat = (
            content_feat - content_mean.expand(size)
        ) / content_std.expand(size)
        return normalized_feat * style_std.view(style_std.shape[0], 1, 1, 1).expand(
            size
        ) + style_mean.view(style_mean.shape[0], 1, 1, 1).expand(size)

    def forward(self, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder module.

        Args:
            c (torch.Tensor): Input tensor (anatomy factors).
            s (torch.Tensor): Style tensor (modality factors).

        Returns:
            torch.Tensor: Decoded output tensor.
        """
        x = c
        for i, f in enumerate(self.layers):
            if (i % 2) == 0:
                x = f(x)
            else:
                x = F.leaky_relu(f(x), 0.2)
                x = Decoder.AdaIN(x, s)
        x = torch.tanh(self.conv(x))
        return x


class Segmentor(ModelBase):
    def __init__(self, parameters, anatomy_factors):
        """
        Segmentor module for SDNet.

        Args:
            parameters (dict): A dictionary containing model parameters.
            anatomy_factors (int): The number of anatomical factors to be considered.

        Attributes:
            layer_list (list): List of layers in the Segmentor module.
            conv (nn.Conv2d): Convolutional layer to generate the final output.
            layers (nn.ModuleList): List of layers in the Segmentor module.
        """

        super(Segmentor, self).__init__(parameters)
        self.layer_list = add_conv_block(
            self.Conv,
            self.BatchNorm,
            in_channels=anatomy_factors,
            out_channels=self.base_filters * 4,
        )
        self.layer_list += add_conv_block(
            self.Conv,
            self.BatchNorm,
            in_channels=self.base_filters * 4,
            out_channels=self.base_filters * 4,
        )
        self.conv = self.Conv(self.base_filters * 4, self.n_classes, 1, 1, 0)
        # Add layers to Module List
        self.layers = nn.ModuleList(self.layer_list)
        self.apply(self.weight_init)

        nn.init.xavier_normal_(self.conv.weight.data)
        self.conv.bias.data.zero_()

    @staticmethod
    def weight_init(model):
        """
        Initialize weights for the given model using He (Kaiming) initialization.

        Args:
            model (nn.Module): Model for which the weights will be initialized.
        """
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Segmentor module.

        Args:
            x (torch.Tensor): Input tensor (anatomy factors).

        Returns:
            torch.Tensor: Segmentation map output tensor.
        """
        for i, f in enumerate(self.layers):
            if (i % 2) == 0:
                x = f(x)
            else:
                x = F.leaky_relu(f(x), 0.2)
        x = F.softmax(self.conv(x), dim=1)
        return x


class ModalityEncoder(ModelBase):
    def __init__(self, parameters, anatomy_factors, modality_factors, num_layers=4):
        """
        Modality Encoder module for SDNet.

        Args:
            parameters (dict): A dictionary containing model parameters.
            anatomy_factors (int): The number of anatomical factors to be considered.
            modality_factors (int): The number of modality factors to be considered.
            num_layers (int, optional): The number of layers in the Modality Encoder. Defaults to 4.

        Attributes:
            num_layers (int): The number of layers in the Modality Encoder.
            layer_list (list): List of layers in the Modality Encoder module.
            fc (nn.Linear): Fully connected layer for the Modality Encoder.
            norm (nn.BatchNorm1d): Batch normalization layer for the Modality Encoder.
            mu_fc (nn.Linear): Fully connected layer for the mean of modality factors.
            logvar_fc (nn.Linear): Fully connected layer for the log variance of modality factors.
            layers (nn.ModuleList): List of layers in the Modality Encoder module.
        """
        super(ModalityEncoder, self).__init__(parameters)
        self.num_layers = num_layers
        self.layer_list = add_downsample_conv_block(
            self.Conv, self.BatchNorm, in_ch=anatomy_factors + 1, out_ch=16
        )
        for _ in range(self.num_layers - 1):
            self.layer_list += add_downsample_conv_block(
                self.Conv, self.BatchNorm, in_ch=16, out_ch=16
            )
        self.fc = nn.Linear(3136, 32)
        self.norm = nn.BatchNorm1d(32)
        self.mu_fc = nn.Linear(32, modality_factors)
        self.logvar_fc = nn.Linear(32, modality_factors)

        # Add layers to Module List
        self.layers = nn.ModuleList(self.layer_list)
        self.apply(self.weight_init)

        nn.init.xavier_normal_(self.fc.weight.data)
        self.fc.bias.data.zero_()
        nn.init.xavier_normal_(self.mu_fc.weight.data)
        self.mu_fc.bias.data.zero_()
        nn.init.xavier_normal_(self.logvar_fc.weight.data)
        self.logvar_fc.bias.data.zero_()

    @staticmethod
    def weight_init(model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Modality Encoder module.

        Args:
            x (torch.Tensor): Input tensor (image data).
            c (torch.Tensor): Anatomy tensor (anatomical factors).

        Returns:
            torch.Tensor: Tuple containing mean (mu) and log variance (logvar) of the modality factors.
        """
        x = torch.cat((x, c), 1)
        for i, f in enumerate(self.layers):
            if (i % 2) == 0:
                x = f(x)
            else:
                x = F.leaky_relu(f(x), 0.2)
        x = self.norm(self.fc(x.view(-1, 3136)))
        x = F.leaky_relu(x, 0.3)
        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)
        return mu, logvar


class SDNet(ModelBase):
    def __init__(self, parameters: dict):
        """
        SDNet (Structure-Disentangled Network) module.

        Args:
            parameters (dict): A dictionary containing model parameters.

        Attributes:
            anatomy_factors (int): The number of anatomical factors to be considered.
            modality_factors (int): The number of modality factors to be considered.
            cencoder (unet): U-Net based Content Encoder for generating anatomy factors.
            mencoder (ModalityEncoder): Modality Encoder for generating modality factors.
            decoder (Decoder): Decoder module for generating the reconstructed image.
            segmentor (Segmentor): Segmentor module for generating the segmentation map.
        """
        super(SDNet, self).__init__(parameters)
        self.anatomy_factors = 8
        self.modality_factors = 8

        if parameters["patch_size"] != [224, 224, 1]:
            print(
                "WARNING: The patch size is not 224x224, which is required for sdnet. Using default patch size instead",
                file=sys.stderr,
            )
            parameters["patch_size"] = [224, 224, 1]

        if parameters["batch_size"] == 1:
            raise ValueError("'batch_size' needs to be greater than 1 for 'sdnet'")

        # amp is not supported for sdnet
        parameters["model"]["amp"] = False
        parameters["model"]["norm_type"] = "instance"

        parameters_unet = deepcopy(parameters)
        parameters_unet["model"]["num_classes"] = self.anatomy_factors
        parameters_unet["model"]["norm_type"] = "instance"
        parameters_unet["model"]["final_layer"] = None

        self.cencoder = unet(parameters_unet)
        self.mencoder = ModalityEncoder(
            parameters, self.anatomy_factors, self.modality_factors
        )
        self.decoder = Decoder(parameters, self.anatomy_factors)
        self.segmentor = Segmentor(parameters, self.anatomy_factors)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from a Gaussian distribution.

        Args:
            mu (torch.Tensor): Mean of the Gaussian distribution.
            logvar (torch.Tensor): Log variance of the Gaussian distribution.

        Returns:
            torch.Tensor: Sampled value from the Gaussian distribution.

        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> typing.List[torch.Tensor]:
        """
        Forward pass of the SDNet module.

        Args:
            x (torch.Tensor): Input tensor (image data).

        Returns:
            typing.List[torch.Tensor]: List containing the segmentation map (sm), reconstructed image (reco), mean (mu),
                                       log variance (logvar), and re-encoded modality factors (modality_factors_reencoded).
        """
        if x.shape[1] > 1:
            x = x[:, 0:1, :, :]
        anatomy_factors = F.gumbel_softmax(self.cencoder(x), hard=True, dim=1)
        mu, logvar = self.mencoder(x, anatomy_factors)
        modality_factors = SDNet.reparameterize(mu, logvar)
        sm = self.segmentor(anatomy_factors)
        reco = self.decoder(anatomy_factors, modality_factors)
        modality_factors_reencoded, _ = self.mencoder(reco, anatomy_factors)
        # sm, anatomy_factors, mu, logvar, modality_factors_reencoded
        return (sm, reco, mu, logvar, modality_factors_reencoded)
