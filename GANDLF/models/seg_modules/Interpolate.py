import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, interp_args=None):
        super(Interpolate, self).__init__()
        if interp_args is None:
            interp_args = {
                "size": None,
                "scale_factor": 2,
                "mode": "nearest",
                "align_corners": True,
            }

        self.interp = nn.functional.interpolate(**interp_args)

    def forward(self, x):
        return self.interp(x)
