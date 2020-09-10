
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelBase(nn.Module):
    '''
    This is the base model class that all other architectures will need to derive from
    '''
    def __init__(self, n_channels, n_classes, base_filters, final_convolution_layer):
        super(ModelBase, self).__init__()
        '''
        This defines all defaults that the model base uses
        '''
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_filters = base_filters

        if final_convolution_layer == 'sigmoid':
            self.final_convolution_layer = F.sigmoid
        elif final_convolution_layer == 'softmax':
            self.final_convolution_layer = F.softmax
        elif (final_convolution_layer == 'none') or (final_convolution_layer == None) or (final_convolution_layer == 'None') or (final_convolution_layer == 'regression'):
            self.final_convolution_layer = None