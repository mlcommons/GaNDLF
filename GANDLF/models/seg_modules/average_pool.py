import torch.nn as nn
import torch.nn.functional as F


class GlobalAveragePooling2D(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling2D, self).__init__()

    def forward(self, x):
        assert len(x.size()) == 4, x.size()
        B, C, W, H = x.size()
        return F.avg_pool2d(x, (W, H)).view(B, C)


class GlobalAveragePooling3D(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling3D, self).__init__()

    def forward(self, x):
        assert len(x.size()) == 5, x.size()
        B, C, W, H, D = x.size()
        return F.avg_pool3d(x, (W, H, D)).view(B, C)
