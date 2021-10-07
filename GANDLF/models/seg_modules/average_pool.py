import torch.nn as nn
import torch.nn.functional as F


class GlobalAveragePooling2D(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling2D, self).__init__()

    def forward(self, x):
        assert len(x.size()) == 4, x.size()
        B, C, W, H = x.size()

        # This is a temporary fix to make sure the size is an integer, not a tensor
        if isinstance(B, int):
           return F.avg_pool2d(x, (W, H)).view(B, C)
        else:
           return F.avg_pool2d(x, (W.item(), H.item())).view(B.item(), C.item())
        

class GlobalAveragePooling3D(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling3D, self).__init__()

    def forward(self, x):
        assert len(x.size()) == 5, x.size()
        B, C, W, H, D = x.size()

        # This is a temporary fix to make sure the size is an integer, not a tensor
        if isinstance(B, int):
           return F.avg_pool3d(x, (W, H, D)).view(B, C)
        else:
           return F.avg_pool2d(x, (W.item(), H.item(), D.item())).view(B.item(), C.item())

