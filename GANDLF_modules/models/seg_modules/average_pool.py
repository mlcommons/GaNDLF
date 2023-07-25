import torch.nn as nn
import torch.nn.functional as F


class GlobalAveragePooling2D(nn.Module):
    """
    Global average pooling operation for 2D tensors.
    """

    def __init__(self):
        super(GlobalAveragePooling2D, self).__init__()

    def forward(self, x):
        """
        Forward pass for global average pooling of a 2D tensor.

        Args:
            x (torch.Tensor):
                Input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Resulting tensor with shape (batch_size, channels).
        """
        assert (
            len(x.size()) == 4
        ), f"Expected 4D input tensor, but got {len(x.size())}D tensor."

        B, C, W, H = x.size()

        # This is a temporary fix to make sure the size is an integer, not a tensor
        if isinstance(B, int):
            return F.avg_pool2d(x, (W, H)).view(B, C)
        else:
            return F.avg_pool2d(x, (W.item(), H.item())).view(B.item(), C.item())


class GlobalAveragePooling3D(nn.Module):
    """
    Global average pooling operation for 3D tensors.
    """

    def __init__(self):
        super(GlobalAveragePooling3D, self).__init__()

    def forward(self, x):
        """
        Forward pass for global average pooling of a 3D tensor.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Resulting tensor with shape (batch_size, channels).
        """
        assert (
            len(x.size()) == 5
        ), f"Expected 5D input tensor, but got {len(x.size())}D tensor."
        B, C, W, H, D = x.size()

        # This is a temporary fix to make sure the size is an integer, not a tensor
        if isinstance(B, int):
            return F.avg_pool3d(x, (W, H, D)).view(B, C)
        else:
            return F.avg_pool3d(x, (W.item(), H.item(), D.item())).view(
                B.item(), C.item()
            )
