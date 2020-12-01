import os
from collections import Counter
import numpy as np

import torch.optim as optim
from GANDLF.schd import *
from GANDLF.models.fcn import fcn
from GANDLF.models.unet import unet
from GANDLF.models.uinc import uinc
from GANDLF.models.MSDNet import MSDNet
from GANDLF.models.densenet import _densenet
from GANDLF.models.vgg import VGG, make_layers, cfg
from GANDLF.losses import *
from GANDLF.utils import *

def get_model(which_model, n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer, psize, **kwargs):
    '''
    This function takes the default constructor and returns the model

    kwargs can be used to pass key word arguments and use arguments that are not explicitly defined.
    '''

    divisibilityCheck_patch = True
    divisibilityCheck_baseFilter = True

    divisibilityCheck_denom_patch = 16 # for unet/resunet/uinc
    divisibilityCheck_denom_baseFilter = 4 # for uinc
    
    if which_model == 'resunet':
        model = unet(n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer = final_convolution_layer, residualConnections=True)
        divisibilityCheck_baseFilter = False
    elif which_model == 'unet':
        model = unet(n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer = final_convolution_layer)
        divisibilityCheck_baseFilter = False
    elif which_model == 'fcn':
        model = fcn(n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer = final_convolution_layer)
        # not enough information to perform checking for this, yet
        divisibilityCheck_patch = False 
        divisibilityCheck_baseFilter = False
    elif which_model == 'uinc':
        model = uinc(n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer = final_convolution_layer)
    elif which_model == 'msdnet':
        model = MSDNet(n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer = final_convolution_layer)
    elif which_model == 'densenet121': # regressor network
        # ref: https://arxiv.org/pdf/1608.06993.pdf
        model = _densenet(n_dimensions, 'densenet121', 32, (6, 12, 24, 16), 64, final_convolution_layer = final_convolution_layer) # are these configurations fine? - taken from torch
    elif which_model == 'densenet161': # regressor network 
        # ref: https://arxiv.org/pdf/1608.06993.pdf
        model = _densenet(n_dimensions, 'densenet161', 48, (6, 12, 36, 24), 96, final_convolution_layer = final_convolution_layer) # are these configurations fine? - taken from torch
    elif which_model == 'densenet169': # regressor network
        # ref: https://arxiv.org/pdf/1608.06993.pdf
        model = _densenet(n_dimensions, 'densenet169', 32, (6, 12, 32, 32), 64, final_convolution_layer = final_convolution_layer) # are these configurations fine? - taken from torch
    elif which_model == 'densenet201': # regressor network
        # ref: https://arxiv.org/pdf/1608.06993.pdf
        model = _densenet(n_dimensions, 'densenet201', 32, (6, 12, 48, 32), 64, final_convolution_layer = final_convolution_layer) # are these configurations fine? - taken from torch
    elif which_model == 'vgg16':
        vgg_config = cfg['D']
        num_final_features = vgg_config[-2]
        divisibility_factor = Counter(vgg_config)['M']
        if psize[-1] == 1:
            psize_altered = np.array(psize[:-1])
        else:
            psize_altered = np.array(psize)

        featuresForClassifier = num_final_features * np.prod(psize_altered // 2**divisibility_factor)
        layers = make_layers(cfg['D'], n_dimensions, n_channels)
        # n_classes is coming from 'class_list' in config, which needs to be changed to use a different variable for regression
        model = VGG(n_dimensions, layers, featuresForClassifier, n_classes, final_convolution_layer = final_convolution_layer)
    else:
        print('WARNING: Could not find the requested model \'' + which_model + '\' in the implementation, using ResUNet, instead', file = sys.stderr)
        which_model = 'resunet'
        model = unet(n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer = final_convolution_layer, residualConnections=True)
    
    # check divisibility
    if divisibilityCheck_patch:
        if not checkPatchDivisibility(psize, divisibilityCheck_denom_patch):
            sys.exit('The \'patch_size\' should be divisible by \'' + str(divisibilityCheck_denom_patch) + '\' for the \'' + which_model + '\' architecture')
    if divisibilityCheck_baseFilter:
        if not checkPatchDivisibility(base_filters, divisibilityCheck_denom_baseFilter):
            sys.exit('The \'base_filters\' should be divisible by \'' + str(divisibilityCheck_denom_baseFilter) + '\' for the \'' + which_model + '\' architecture')
    
    return model

def get_loss(which_loss):
    '''
    This function parses the loss coming from the config file and returns the appropriate object
    '''
    MSE_requested = False
    if isinstance(which_loss, dict): # this is currently only happening for mse_torch
        # check for mse_torch
        loss_fn = MSE_loss
        MSE_requested = True
    else: # this is a simple string, so proceed with previous workflow
        MSE_requested = False
        if which_loss == 'dc':
            loss_fn = MCD_loss
        elif which_loss == 'dcce':
            loss_fn = DCCE
        elif which_loss == 'ce':
            loss_fn = CE
        # elif loss_function == 'mse':
        #     loss_fn = MCD_MSE_loss
        else:
            print('WARNING: Could not find the requested loss function \'' + loss_fn + '\' in the implementation, using dc, instead', file = sys.stderr)
            which_loss = 'dc'
            loss_fn = MCD_loss

    return loss_fn, MSE_requested

def get_optimizer(which_optimizer, model_parameters, learning_rate):
    '''
    This function parses the optimizer from the config file and returns the appropriate object
    '''
    if which_optimizer == 'sgd':
        optimizer = optim.SGD(model_parameters,
                              lr=learning_rate,
                              momentum = 0.9)
    elif which_optimizer == 'adam':        
        optimizer = optim.Adam(model_parameters,
                               lr=learning_rate,
                               betas = (0.9,0.999),
                               weight_decay = 0.00005)
    else:
        print('WARNING: Could not find the requested optimizer \'' + which_optimizer + '\' in the implementation, using sgd, instead', file = sys.stderr)
        opt = 'sgd'
        optimizer = optim.SGD(model_parameters,
                              lr= learning_rate,
                              momentum = 0.9)

    return optimizer

def get_scheduler(which_scheduler, optimizer, batch_size, training_samples_size, learning_rate):
    '''
    This function parses the optimizer from the config file and returns the appropriate object
    '''
    step_size = 4*batch_size*training_samples_size
    if which_scheduler == "triangle":
        clr = cyclical_lr(step_size, min_lr = 10**-3, max_lr=1)
        scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
        print("Initial Learning Rate: ",learning_rate)
    elif which_scheduler == "exp":
        scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif which_scheduler == "step":
        scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    elif which_scheduler == "reduce-on-plateau":
        scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                                  patience=10, threshold=0.0001, threshold_mode='rel',
                                                                  cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    elif which_scheduler == "triangular":
        scheduler_lr = torch.optim.lr_scheduler.CyclicLR(optimizer, learning_rate * 0.001, learning_rate,
                                                         step_size_up=step_size,
                                                         step_size_down=None, mode='triangular', gamma=1.0,
                                                         scale_fn=None, scale_mode='cycle', cycle_momentum=True,
                                                         base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
    else:
        print('WARNING: Could not find the requested Learning Rate scheduler \'' + which_scheduler + '\' in the implementation, using exp, instead', file=sys.stderr)
        scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)

    return scheduler_lr
