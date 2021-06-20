import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, interp_kwargs=None):
        nn.Module.__init__(self)
        if interp_kwargs is None:
            self.interp_kwargs = {
                "size": None,
                "scale_factor": 2,
                "mode": "bilinear",
                "align_corners": True,
            }

        self.interp_kwargs = interp_kwargs

    def forward(self, x):
        return nn.functional.interpolate(x, **(self.interp_kwargs))
