import torch
import torch.nn as nn

from ..modelBase import ModelBase

class vanillaGANDiscriminator(ModelBase):
    def __init__(
        self,
        parameters
    ):
        super(vanillaGANDiscriminator, self).__init__(parameters)
        self.image_dim = parameters["patch_size"][0]
        self.model = nn.Sequential(
            nn.Linear(parameters["model"]["num_channels"] * self.image_dim  * self.image_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def weight_init(model):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        if len(x.shape) > 4:
            x = x.permute(0,1,4,2,3) #Sarthak
            x_part = x[:,0].clone()
        else:
            x_part = x.clone()
        x_flat = x_part.view(x_part.shape[0], -1)
        out = self.model(x_flat)

        return out