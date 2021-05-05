import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAveragePooing, self).__init__()
        
    def forward(self, x):
        assert len(x.size()) == 4, x.size()
        B, C, W, H = x.size()
        return F.avg_pool2d(x, (W, H)).view(B, C)


class GlobalAvgPool3d(nn.Module):
    def __init__(self):
        super(GlobalAveragePooing, self).__init__()
        
    def forward(self, x):
        assert len(x.size()) == 5, x.size()
        B, C, X, Y, Z = x.size()
        return F.avg_pool3d(x, (X, Y, Z)).view(B, C)
