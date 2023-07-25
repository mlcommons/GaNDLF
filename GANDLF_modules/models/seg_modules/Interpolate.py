import torch.nn as nn


class Interpolate(nn.Module):
    def __init__(self, interp_kwargs=None):
        """
        Initialize the function.

        Args:
            interp_kwargs (list, optional): Keyword arguments for initialization. Defaults to None.
        """
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
