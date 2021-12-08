import torch
import torch.nn as nn
import numpy as np

from ..modelBase import ModelBase

class vanillaGANGenerator(ModelBase):
    def __init__(
        self,
        parameters
    ):
        super(vanillaGANGenerator, self).__init__(parameters)
        self.image_shape = torch.Size([parameters["model"]["num_channels"], parameters["patch_size"][0], parameters["patch_size"][0]])
        self.latent_dim = parameters["latent_dim"]

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # self.first_block = block(self.latent_dim, 128, normalize=False)[0]
        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.image_shape))),
            nn.Tanh()
        )

    @staticmethod
    def weight_init(model):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, z):
        out = self.model(z) #@spthermo need to update it for batch_size == 1
        out = out.view(z.shape[0], self.image_shape[0], self.image_shape[1], self.image_shape[2])
        return out