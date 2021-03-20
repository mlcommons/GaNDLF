from collections import Counter
import numpy as np

import torch.optim as optim
from torch.optim.lr_scheduler import *
from GANDLF.schd import *
from GANDLF.models.fcn import fcn
from GANDLF.models.unet import unet
from GANDLF.models.uinc import uinc
from GANDLF.models.MSDNet import MSDNet
from GANDLF.models import densenet
from GANDLF.models.vgg import VGG, make_layers, cfg
from GANDLF.losses import *
from GANDLF.utils import *
import torchvision
import torch.nn as nn

def get_model(which_model, n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer, psize, batch_size, **kwargs):
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
    elif 'densenet' in which_model: # common parsing for densenet
        if which_model == 'densenet121':
            model = densenet.generate_model(model_depth=121,
                                            num_classes=n_classes,
                                            n_dimensions=n_dimensions,
                                            n_input_channels=n_channels, final_convolution_layer = final_convolution_layer)
        elif which_model == 'densenet161': # regressor/classifier network
            model = densenet.generate_model(model_depth=161,
                                            num_classes=n_classes,
                                            n_dimensions=n_dimensions,
                                            n_input_channels=n_channels, final_convolution_layer = final_convolution_layer)
        elif which_model == 'densenet169': # regressor/classifier network
            model = densenet.generate_model(model_depth=169,
                                            num_classes=n_classes,
                                            n_dimensions=n_dimensions,
                                            n_input_channels=n_channels, final_convolution_layer = final_convolution_layer)
        elif which_model == 'densenet201': # regressor/classifier network
            model = densenet.generate_model(model_depth=201,
                                            num_classes=n_classes,
                                            n_dimensions=n_dimensions,
                                            n_input_channels=n_channels, final_convolution_layer = final_convolution_layer)
        elif which_model == 'densenet264': # regressor/classifier network
            model = densenet.generate_model(model_depth=264,
                                            num_classes=n_classes,
                                            n_dimensions=n_dimensions,
                                            n_input_channels=n_channels, final_convolution_layer = final_convolution_layer)
        else:
            sys.exit('Requested DENSENET type \'' + which_model + '\' has not been implemented')
    elif 'vgg' in which_model: # common parsing for vgg
        if which_model == 'vgg11':
            vgg_config = cfg['A']
        elif which_model == 'vgg13':
            vgg_config = cfg['B']
        elif which_model == 'vgg16':
            vgg_config = cfg['D']
        elif which_model == 'vgg19':
            vgg_config = cfg['E']
        else:
            sys.exit('Requested VGG type \'' + which_model + '\' has not been implemented')

        if 'batch_norm' in kwargs:
            batch_norm = kwargs.get("batch_norm")
        else:
            batch_norm = True
        num_final_features = vgg_config[-2]
        m_counter = Counter(vgg_config)['M']
        if psize[-1] == 1:
            psize_altered = np.array(psize[:-1])
        else:
            psize_altered = np.array(psize)
        divisibilityCheck_patch = False 
        divisibilityCheck_baseFilter = False
        featuresForClassifier = batch_size * num_final_features * np.prod(psize_altered // 2**m_counter)
        layers = make_layers(vgg_config, n_dimensions, n_channels, batch_norm=batch_norm)
        # n_classes is coming from 'class_list' in config, which needs to be changed to use a different variable for regression
        model = VGG(n_dimensions, layers, featuresForClassifier, n_classes, final_convolution_layer = final_convolution_layer)
    elif which_model == 'brain_age':
        if n_dimensions != 2:
            sys.exit("Brain Age predictions only works on 2D data")
        model = torchvision.models.vgg16(pretrained = True)
        model.final_convolution_layer = None
        # Freeze training for all layers
        for param in model.features.parameters():
            param.require_grad = False
        # Newly created modules have require_grad=True by default
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1] # Remove last layer
        #features.extend([nn.AvgPool2d(1024), nn.Linear(num_features,1024),nn.ReLU(True), nn.Dropout2d(0.8), nn.Linear(1024,1)]) # RuntimeError: non-empty 2D or 3D (batch mode) tensor expected for input
        features.extend([nn.Linear(num_features,1024),nn.ReLU(True), nn.Dropout2d(0.8), nn.Linear(1024,1)])
        model.classifier = nn.Sequential(*features) # Replace the model classifier
        divisibilityCheck_patch = False 
        divisibilityCheck_baseFilter = False
        
    else:
        print('WARNING: Could not find the requested model \'' + which_model + '\' in the implementation, using ResUNet, instead', file = sys.stderr)
        which_model = 'resunet'
        model = unet(n_dimensions, n_channels, n_classes, base_filters, final_convolution_layer = final_convolution_layer, residualConnections=True)
    
    # check divisibility
    if divisibilityCheck_patch:
        if not checkPatchDivisibility(psize, divisibilityCheck_denom_patch):
            sys.exit('The \'patch_size\' should be divisible by \'' + str(divisibilityCheck_denom_patch) + '\' for the \'' + which_model + '\' architecture')
    if divisibilityCheck_baseFilter:
        if base_filters % divisibilityCheck_denom_baseFilter != 0:
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
        elif which_loss == 'dc_log':
            loss_fn = MCD_log_loss
        elif which_loss == 'dcce':
            loss_fn = DCCE
        elif which_loss == 'ce':
            loss_fn = CE
        # elif loss_function == 'mse':
        #     loss_fn = MCD_MSE_loss
        else:
            print('WARNING: Could not find the requested loss function \'' + which_loss + '\' in the implementation, using dc, instead', file = sys.stderr)
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
        scheduler_lr = LambdaLR(optimizer, [clr])
        print("Initial Learning Rate: ",learning_rate)
    elif which_scheduler == "triangle_modified":
        step_size = training_samples_size/learning_rate
        clr = cyclical_lr_modified(step_size)
        scheduler_lr = LambdaLR(optimizer, [clr])
        print("Initial Learning Rate: ",learning_rate)
    elif which_scheduler == "exp":
        scheduler_lr = ExponentialLR(optimizer, learning_rate, last_epoch=-1)
    elif which_scheduler == "step":
        scheduler_lr = StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    elif which_scheduler == "reduce-on-plateau":
        scheduler_lr = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                        patience=10, threshold=0.0001, threshold_mode='rel',
                                        cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    elif which_scheduler == "triangular":
        scheduler_lr = CyclicLR(optimizer, learning_rate * 0.001, learning_rate,
                                step_size_up=step_size,
                                step_size_down=None, mode='triangular', gamma=1.0,
                                scale_fn=None, scale_mode='cycle', cycle_momentum=True,
                                base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
    elif which_scheduler == 'cosineannealing':
        scheduler_lr = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6, last_epoch=-1)
    else:
        print('WARNING: Could not find the requested Learning Rate scheduler \'' + which_scheduler + '\' in the implementation, using exp, instead', file=sys.stderr)
        scheduler_lr = ExponentialLR(optimizer, 0.1, last_epoch=-1)

    return scheduler_lr

    """
    # initialize without considering background
    dice_weights_dict = {} # average for "weighted averaging"
    dice_penalty_dict = {} # penalty for misclassification
    for i in range(1, n_classList):
        dice_weights_dict[i] = 0
        dice_penalty_dict[i] = 0

    # define a seaparate data loader for penalty calculations
    penaltyData = ImagesFromDataFrame(trainingDataFromPickle, psize, headers, q_max_length, q_samples_per_volume, q_num_workers, q_verbose, sampler = parameters['patch_sampler'], train=False, augmentations=None,preprocessing=preprocessing) 
    penalty_loader = DataLoader(penaltyData, batch_size=batch_size, shuffle=True)
    
    # get the weights for use for dice loss
    total_nonZeroVoxels = 0
    for batch_idx, (subject) in enumerate(penalty_loader): # iterate through full training data
        # accumulate dice weights for each label
        mask = subject['label'][torchio.DATA]
        one_hot_mask = one_hot(mask, class_list)
        for i in range(1, n_classList):
            currentNumber = torch.nonzero(one_hot_mask[:,i,:,:,:], as_tuple=False).size(0)
            dice_weights_dict[i] = dice_weights_dict[i] + currentNumber # class-specific non-zero voxels
            total_nonZeroVoxels = total_nonZeroVoxels + currentNumber # total number of non-zero voxels to be considered
    
    # get the penalty values - dice_weights contains the overall number for each class in the training data
    for i in range(1, n_classList):
        penalty = total_nonZeroVoxels # start with the assumption that all the non-zero voxels make up the penalty
        for j in range(1, n_classList):
            if i != j: # for differing classes, subtract the number
                penalty = penalty - dice_penalty_dict[j]
        
        dice_penalty_dict[i] = penalty / total_nonZeroVoxels # this is to be used to weight the loss function
    dice_weights_dict[i] = 1 - dice_weights_dict[i]# this can be used for weighted averaging
    """
