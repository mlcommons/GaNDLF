import torch.nn as nn

from ..modelBase import ModelBase

class DCGANGenerator(ModelBase):
    def __init__(
        self,
        parameters
    ):
        super(DCGANGenerator, self).__init__(parameters)

        self.init_size = parameters["patch_size"][0] // 4
        self.l1 = nn.Sequential(nn.Linear(parameters["latent_dim"], 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, parameters["model"]["num_channels"], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out = self.conv_blocks(out)
        
        return out