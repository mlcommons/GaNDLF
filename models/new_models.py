import torch.nn.functional as F
import torch.nn as nn
import torch
from seg_modules import in_conv, DownsamplingModule, EncodingModule, InceptionModule, ResNetModule
from seg_modules import UpsamplingModule, DecodingModule,IncDownsamplingModule,IncConv
from seg_modules import out_conv, FCNUpsamplingModule, IncDropout,IncUpsamplingModule

"""
The smaller individual modules of these networks (the ones defined below), are taken from the seg_modules files as seen in the imports above.

"""






