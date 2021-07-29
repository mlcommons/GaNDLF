import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

from GANDLF.models.seg_modules.add_conv_block import add_conv_block
from GANDLF.models.seg_modules.add_downsample_conv_block import (
    add_downsample_conv_block,
)
from GANDLF.models.unet import unet
from .modelBase import ModelBase


class Decoder(ModelBase):
    def __init__(
        self,
        n_dimensions,
        n_channels,
        n_classes,
        base_filters,
        norm_type,
        final_convolution,
        anatomy_factors,
        num_layers=5,
    ):
        super(Decoder, self).__init__(
            n_dimensions,
            n_channels,
            n_classes,
            base_filters,
            norm_type,
            final_convolution,
        )
        self.num_layers = num_layers
        self.layer_list = add_conv_block(
            self.Conv, self.BatchNorm, in_ch=anatomy_factors, out_ch=base_filters
        )
        for _ in range(self.num_layers - 2):
            self.layer_list += add_conv_block(
                self.Conv, self.BatchNorm, in_ch=base_filters, out_ch=base_filters
            )
        self.conv = self.Conv(base_filters, 1, 3, 1, 1)
        # Add layers to Module List
        self.layers = nn.ModuleList(self.layer_list)
        self.apply(self.weight_init)

        nn.init.xavier_normal_(self.conv.weight.data)
        self.conv.bias.data.zero_()

    @staticmethod
    def CalcVectorMeanStd(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        feat_var = feat.var(dim=1) + eps
        feat_std = feat_var.sqrt()
        feat_mean = feat.mean(dim=1)
        return feat_mean, feat_std

    @staticmethod
    def CalcTensorMeanStd(feat, eps=1e-5):
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
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    @staticmethod
    def AdaIN(content_feat, style_feat):
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
    def __init__(
        self,
        n_dimensions,
        n_channels,
        n_classes,
        base_filters,
        norm_type,
        final_convolution,
        anatomy_factors,
    ):
        super(Segmentor, self).__init__(
            n_dimensions,
            n_channels,
            n_classes,
            base_filters,
            norm_type,
            final_convolution,
        )
        self.layer_list = add_conv_block(
            self.Conv, self.BatchNorm, in_ch=anatomy_factors, out_ch=base_filters * 4
        )
        self.layer_list += add_conv_block(
            self.Conv, self.BatchNorm, in_ch=base_filters * 4, out_ch=base_filters * 4
        )
        self.conv = self.Conv(base_filters * 4, n_classes, 1, 1, 0)
        # Add layers to Module List
        self.layers = nn.ModuleList(self.layer_list)
        self.apply(self.weight_init)

        nn.init.xavier_normal_(self.conv.weight.data)
        self.conv.bias.data.zero_()

    @staticmethod
    def weight_init(model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, f in enumerate(self.layers):
            if (i % 2) == 0:
                x = f(x)
            else:
                x = F.leaky_relu(f(x), 0.2)
        x = F.softmax(self.conv(x), dim=1)
        return x


class ModalityEncoder(ModelBase):
    def __init__(
        self,
        n_dimensions,
        n_channels,
        n_classes,
        base_filters,
        norm_type,
        final_convolution,
        anatomy_factors,
        modality_factors,
        num_layers=4,
    ):
        super(ModalityEncoder, self).__init__(
            n_dimensions,
            n_channels,
            n_classes,
            base_filters,
            norm_type,
            final_convolution,
        )
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
    def __init__(
        self,
        n_dimensions,
        n_channels,
        n_classes,
        base_filters,
        norm_type,
        final_convolution_layer,
    ):
        super().__init__(
            n_dimensions,
            n_channels,
            n_classes,
            base_filters,
            norm_type,
            final_convolution_layer,
        )
        self.anatomy_factors = 8
        self.modality_factors = 8

        self.cencoder = unet(
            n_dimensions,
            n_channels,
            self.anatomy_factors,
            base_filters,
            "instance",
            None,
        )
        self.mencoder = ModalityEncoder(
            n_dimensions,
            n_channels,
            n_classes,
            base_filters,
            norm_type,
            final_convolution_layer,
            self.anatomy_factors,
            self.modality_factors,
        )
        self.decoder = Decoder(
            n_dimensions,
            n_channels,
            n_classes,
            base_filters,
            norm_type,
            final_convolution_layer,
            self.anatomy_factors,
        )
        self.segmentor = Segmentor(
            n_dimensions,
            n_channels,
            n_classes,
            base_filters,
            norm_type,
            final_convolution_layer,
            self.anatomy_factors,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> typing.List[torch.Tensor]:
        if x.shape[1] > 1:
            x = x[:, 0:1, :, :]
        anatomy_factors = F.gumbel_softmax(self.cencoder(x), hard=True, dim=1)
        mu, logvar = self.mencoder(x, anatomy_factors)
        modality_factors = SDNet.reparameterize(mu, logvar)
        sm = self.segmentor(anatomy_factors)
        reco = self.decoder(anatomy_factors, modality_factors)
        modality_factors_reencoded, _ = self.mencoder(reco, anatomy_factors)
        return (
            sm,
            reco,
            mu,
            logvar,
            modality_factors_reencoded,
        )  # sm, anatomy_factors, mu, logvar, modality_factors_reencoded
