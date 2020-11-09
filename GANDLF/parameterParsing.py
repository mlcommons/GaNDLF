import torch.optim as optim

from GANDLF.schd import *
from GANDLF.models.fcn import fcn
from GANDLF.models.unet import unet
from GANDLF.models.uinc import uinc
from GANDLF.losses import *
from GANDLF.utils import *

def get_model(which_model, n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer, **kwargs):
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
    # setting optimizer
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