import torch.nn as nn

from ..modelBase import ModelBase

class LSGANDiscriminator(ModelBase):
    def __init__(
        self,
        parameters
    ):
        super(LSGANDiscriminator, self).__init__(parameters)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(parameters["model"]["num_channels"], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = parameters["patch_size"][0] // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, x):
        if len(x.shape) > 4:
            x = x.permute(0,1,4,2,3) #Sarthak
            x_part = x[:,0].clone()
        else:
            x_part = x.clone()
        x = self.model(x_part)
        x = x.view(x.shape[0], -1)
        out = self.adv_layer(x)

        return out