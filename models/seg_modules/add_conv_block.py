import torch.nn.functional as F
import torch.nn as nn
import torch

def add_conv_block(in_ch=1, out_ch=1, kernel_size=3, dilate=1, last=False, volumetric=True):
    """
    Helpder function
    """
    if volumetric:
        Conv = nn.Conv3d
        BatchNorm = nn.BatchNorm3d
    else:
        Conv = nn.Conv2d
        BatchNorm = nn.BatchNorm2d
    pad = dilate if not last else 0
    conv_1 = Conv(in_ch, out_ch, kernel_size, padding=pad, dilation=dilate)
    bn_1 = BatchNorm(out_ch)

    return [conv_1, bn_1]