import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch.utils.data.dataset import Dataset
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import random
# import scipy
import torchio
from torchio.transforms import *
from torchio import Image, Subject
from sklearn.model_selection import KFold
from shutil import copyfile
import time
import sys
import pickle
from pathlib import Path
import argparse
import datetime
import SimpleITK as sitk
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.schd import *
from GANDLF.models.fcn import fcn
from GANDLF.models.unet import unet
from GANDLF.models.resunet import resunet
from GANDLF.models.uinc import uinc
from GANDLF.losses import *
from GANDLF.utils import *

def trainingLoop(trainingDataFromPickle, validataionDataFromPickle, headers, device, parameters, outputDir):
    '''
    This is the main training loop
    '''
    # extract variables form parameters dict
    psize = parameters['psize']
    q_max_length = parameters['q_max_length']
    q_samples_per_volume = parameters['q_samples_per_volume']
    q_num_workers = parameters['q_num_workers']
    q_verbose = parameters['q_verbose']
    augmentations = parameters['data_augmentation']
    which_model = parameters['model']['architecture']
    opt = parameters['opt']
    loss_function = parameters['loss_function']
    scheduler = parameters['scheduler']
    class_list = parameters['class_list']
    base_filters = parameters['base_filters']
    batch_size = parameters['batch_size']
    learning_rate = parameters['learning_rate']
    num_epochs = parameters['num_epochs']
    amp = parameters['amp']
    patience = parameters['patience']
    n_channels = len(headers['channelHeaders'])
    n_classList = len(class_list)
  
    holdoutDataFromPickle = pd.read_csv("./csvs/test_e1.csv")
    trainingDataForTorch = ImagesFromDataFrame(trainingDataFromPickle, psize, headers, q_max_length, q_samples_per_volume,
                                               q_num_workers, q_verbose, train=True, augmentations=augmentations, resize = parameters['resize'])
    validationDataForTorch = ImagesFromDataFrame(validataionDataFromPickle, psize, headers, q_max_length, q_samples_per_volume,
                                               q_num_workers, q_verbose, train=True, augmentations=augmentations, resize = parameters['resize']) # may or may not need to add augmentations here
    inferenceDataForTorch = ImagesFromDataFrame(holdoutDataFromPickle, psize, headers, q_max_length, q_samples_per_volume,
                                            q_num_workers, q_verbose, train=False, augmentations=augmentations, resize = parameters['resize'])
    
    
    train_loader = DataLoader(trainingDataForTorch, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validationDataForTorch, batch_size=1)
    inference_loader = DataLoader(inferenceDataForTorch,batch_size=1)
    
    # sanity check
    if n_channels == 0:
        sys.exit('The number of input channels cannot be zero, please check training CSV')

    # Defining our model here according to parameters mentioned in the configuration file : 
    if which_model == 'resunet':
        model = resunet(n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])
    elif which_model == 'unet':
        model = unet(n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])
    elif which_model == 'fcn':
        model = fcn(n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])
    elif which_model == 'uinc':
        model = uinc(n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])
    else:
        print('WARNING: Could not find the requested model \'' + which_model + '\' in the implementation, using ResUNet, instead', file = sys.stderr)
        which_model = 'resunet'
        model = resunet(n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])

    # setting optimizer
    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate,
                              momentum = 0.9)
    elif opt == 'adam':        
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate,
                               betas = (0.9,0.999),
                               weight_decay = 0.00005)
    else:
        print('WARNING: Could not find the requested optimizer \'' + opt + '\' in the implementation, using sgd, instead', file = sys.stderr)
        opt = 'sgd'
        optimizer = optim.SGD(model.parameters(),
                              lr= learning_rate,
                              momentum = 0.9)
    # setting the loss function
    if isinstance(loss_function, dict): # this is currently only happening for mse_torch
        # check for mse_torch
        loss_fn = MSE_loss
        MSE_requested = True
    else: # this is a simple string, so proceed with previous workflow
        MSE_requested = False
        if loss_function == 'dc':
            loss_fn = MCD_loss
        elif loss_function == 'dcce':
            loss_fn = DCCE
        elif loss_function == 'ce':
            loss_fn = CE
        # elif loss_function == 'mse':
        #     loss_fn = MCD_MSE_loss
        else:
            print('WARNING: Could not find the requested loss function \'' + loss_fn + '\' in the implementation, using dc, instead', file = sys.stderr)
            loss_function = 'dc'
            loss_fn = MCD_loss

    # training_start_time = time.asctime()
    # startstamp = time.time()
    print("\nHostname     :" + str(os.getenv("HOSTNAME")))
    sys.stdout.flush()

    # resume if compatible model was found
    if os.path.exists(os.path.join(outputDir,str(which_model) + "_best.pth.tar")):
        checkpoint = torch.load(os.path.join(outputDir,str(which_model) + "_best.pth.tar"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model checkpoint found. Loading checkpoint from: ",os.path.join(outputDir,str(which_model) + "_best.pth.tar"))

    print("Training Data Samples: ", len(train_loader.dataset))
    sys.stdout.flush()
    if device != 'cpu':
        if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
            sys.exit('Please set the environment variable \'CUDA_VISIBLE_DEVICES\' correctly before trying to run GANDLF on GPU')
        
        dev = os.environ.get('CUDA_VISIBLE_DEVICES')
        # multi-gpu support
        # ###
        # # https://discuss.pytorch.org/t/cuda-visible-devices-make-gpu-disappear/21439/17?u=sarthakpati
        # ###
        if ',' in dev:
            device = torch.device('cuda')
            model = nn.DataParallel(model, '[' + dev + ']')
        else:
            print('Device requested via CUDA_VISIBLE_DEVICES: ', dev)
            if (torch.cuda.device_count() == 1) and (int(dev) == 1): # this should be properly fixed
                dev = '0'
            print('Device finally used: ', dev)
            device = torch.device('cuda:' + dev)
            model = model.to(int(dev))
            print('Memory Total : ', round(torch.cuda.get_device_properties(int(dev)).total_memory/1024**3, 1), 'GB')
            print('Memory Usage : ')
            print('Allocated : ', round(torch.cuda.memory_allocated(int(dev))/1024**3, 1),'GB')
            print('Cached: ', round(torch.cuda.memory_reserved(int(dev))/1024**3, 1), 'GB')
        
        print("Current Device : ", torch.cuda.current_device())
        print("Device Count on Machine : ", torch.cuda.device_count())
        print("Device Name : ", torch.cuda.get_device_name(device))
        print("Cuda Availability : ", torch.cuda.is_available())
        
        # ensuring optimizer is in correct device - https://github.com/pytorch/pytorch/issues/8741
        optimizer.load_state_dict(optimizer.state_dict())

    else:
        dev = -1
        device = torch.device('cpu')
        model.cpu()
        amp = False
        print("Since Device is CPU, Mixed Precision Training is set to False")
        
    print('Using device:', device)
    sys.stdout.flush()

    # Checking for the learning rate scheduler
    if scheduler == "triangle":
        step_size = 4*batch_size*len(train_loader.dataset)
        clr = cyclical_lr(step_size, min_lr = 10**-3, max_lr=1)
        scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
        print("Starting Learning rate is:",learning_rate)
    elif scheduler == "exp":
        scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif scheduler == "step":
        scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    elif scheduler == "reduce-on-plateau":
        scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                                  patience=10, threshold=0.0001, threshold_mode='rel',
                                                                  cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    elif scheduler == "triangular":
        scheduler_lr = torch.optim.lr_scheduler.CyclicLR(optimizer, learning_rate * 0.001, learning_rate,
                                                         step_size_up=4*batch_size*len(train_loader.dataset),
                                                         step_size_down=None, mode='triangular', gamma=1.0,
                                                         scale_fn=None, scale_mode='cycle', cycle_momentum=True,
                                                         base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
    else:
        print('WARNING: Could not find the requested Learning Rate scheduler \'' + scheduler + '\' in the implementation, using exp, instead', file=sys.stderr)
        scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)

    sys.stdout.flush()
    ############## STORING THE HISTORY OF THE LOSSES #################
    best_val_dice = -1
    best_train_dice = -1
    best_test_dice = -1
    total_train_loss = 0
    total_train_dice = 0
    patience_count = 0
    # Creating a CSV to log training loop and writing the initial columns
    log_train_file = os.path.join(outputDir,"trainingScores_log.csv")
    log_train = open(log_train_file,"w")
    log_train.write("Epoch,Train_Loss,Train_Dice, Val_Loss, Val_Dice\n")
    log_train.close()
                                
                                
    # Getting the channels for training and removing all the non numeric entries from the channels
    batch = next(iter(train_loader))
    channel_keys = list(batch.keys())
    channel_keys_new = []

    # automatic mixed precision - https://pytorch.org/docs/stable/amp.html
    if amp:
        scaler = torch.cuda.amp.GradScaler() 

    for item in channel_keys:
        if item.isnumeric():
            channel_keys_new.append(item)
    channel_keys = channel_keys_new
    ################ TRAINING THE MODEL##############
    for ep in range(num_epochs):
        start = time.time()
        print("\n")
        print("Ep # %s LR: %s Start: %s "%(str(ep), str(optimizer.param_groups[0]['lr']), str(datetime.datetime.now())))
        model.train()
        for batch_idx, (subject) in enumerate(train_loader):
            # uncomment line to debug memory issues
            # # print('=== Memory (allocated; cached) : ', round(torch.cuda.memory_allocated(int(dev))/1024**3, 1), '; ', round(torch.cuda.memory_reserved(int(dev))/1024**3, 1))
            # Load the subject and its ground truth
            # read and concat the images
            image = torch.cat([subject[key][torchio.DATA] for key in channel_keys], dim=1) # concatenate channels 
            # read the mask
            mask = subject['label'][torchio.DATA] # get the label image
            # Why are we doing this? Please check again
            #mask = one_hot(mask.cpu().float().numpy(), class_list)
            mask = one_hot(mask, class_list)
            # one_hot_mask = one_hot_mask.unsqueeze(0)
            #mask = torch.from_numpy(mask)
            # Loading images into the GPU and ignoring the affine
            image, mask = image.float().cuda(), mask.cuda()
            # Making sure that the optimizer has been reset
            optimizer.zero_grad()
            # Forward Propagation to get the output from the models
            # TODO: Not recommended? (https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/6)will try without
            # might help solve OOM
            # torch.cuda.empty_cache()
            # Casts operations to mixed precision
            output = model(image)
            if amp:
                with torch.cuda.amp.autocast(): 
                # Computing the loss
                    if MSE_requested:
                        loss = loss_fn(output.double(), mask.double(), n_classList, reduction = loss_function['mse']['reduction'])
                    else:
                        loss = loss_fn(output.double(), mask.double(), n_classList)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
            else:
                # Computing the loss
                if MSE_requested:
                    loss = loss_fn(output.double(), mask.double(), n_classList, reduction = loss_function['mse']['reduction'])
                else:
                    loss = loss_fn(output.double(), mask.double(), n_classList)
                loss.backward()
                optimizer.step()
                           
            #Pushing the dice to the cpu and only taking its value
            curr_loss = loss.cpu().data.item()
            #train_loss_list.append(loss.cpu().data.item())
            total_train_loss += curr_loss
            #Computing the dice score  # Can be changed for multi-class outputs later.
            curr_dice = MCD(output.double(), mask.double(), n_classList).cpu().data.item()
            #print(curr_dice)
            #Computng the total dice
            total_train_dice += curr_dice
            # update scale for next iteration
            if amp:
                scaler.update() 
            # TODO: Not recommended? (https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/6)will try without
            # torch.cuda.empty_cache()
            if scheduler == "triangular":
                scheduler_lr.step()            
            print(curr_dice)
            break

        average_train_dice = total_train_dice/len(train_loader.dataset)
        average_train_loss = total_train_loss/len(train_loader.dataset)
                               
        if average_train_dice > best_train_dice:
            best_train_idx = ep
            best_train_dice = average_train_dice
            torch.save({"epoch": best_train_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_train_dice": best_train_dice }, os.path.join(outputDir, which_model + "_best_train.pth.tar"))

        print("Ep Train DCE: %s Best Train DCE: %s Avg Train Loss: %s Best Train Ep %s"%(str(average_train_dice), str(best_train_dice), str(average_train_loss), str(best_train_idx)))

        # Now we enter the evaluation/validation part of the epoch        
        model.eval()

        # validation data scores
        total_val_dice, total_val_loss = test(model,validationDataForTorch,psize,channel_keys,class_list,loss_fn)
        average_val_dice = total_val_dice/len(val_loader.dataset)
        average_val_loss = total_val_loss/len(val_loader.dataset)

        # testing data scores
        total_test_dice, total_test_loss = test(model,inferenceDataForTorch,psize,channel_keys,class_list,loss_fn) 
        average_test_dice = total_test_dice/len(inference_loader.dataset)
        average_test_loss = total_test_loss/len(inference_loader.dataset)

        if average_val_dice > best_val_dice:
            best_val_idx = ep
            best_val_dice = average_val_dice
            best_test_val_dice = average_test_dice
            # We can add more stuff to be saved if we need anything more
            torch.save({"epoch": best_val_idx,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_dice": best_val_dice }, os.path.join(outputDir, which_model + "_best_val.pth.tar"))
        else:
            patience_count = patience_count + 1 
        print("Ep Val DCE: %s Best Val DCE: %s Avg Val Loss: %s Best Val Ep"%(average_val_dice, best_val_dice, average_val_loss, best_val_idx)) 
        print("Best Test Dice w.r.t val model: ", best_test_val_dice )

        if average_test_dice > best_test_dice:
            best_test_dice = average_test_dice
            best_test_idx = ep 
            torch.save({"epoch": best_test_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_test_dice": best_test_dice }, os.path.join(outputDir, which_model + "_best_test.pth.tar"))
        print("Epoch Test dice:" , average_test_dice)
        print("Best Test Dice:", best_test_dice)


        # Updating the learning rate according to some conditions - reduce lr on plateau needs out loss to be monitored and schedules the LR accordingly. Others change irrespective of loss.
        if not scheduler == "triangular":
            if scheduler == "reduce-on-plateau":
                scheduler_lr.step(average_val_loss)
            else:
                scheduler_lr.step()

        # Saving the current model
        torch.save({"epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": average_val_dice }, os.path.join(outputDir, which_model + "_latest.pth.tar"))

        # Checking if patience is crossed
        if patience_count > patience:
            print("Performance Metric has not improved for %d epochs, exiting training loop"%(patience))
            break
        
        stop = time.time()     
        print("Time for epoch:",(stop - start)/60,"mins")        

        sys.stdout.flush()
        log_train = open(log_train_file, "a")
        log_train.write(str(ep) + "," + str(average_train_loss) + "," + str(average_train_dice) + "," + str(average_val_loss) + "," + str(average_val_dice) + "," + str(average_test_dice) + "\n")
        log_train.close()
        total_test_dice = 0
        total_test_loss = 0
        total_train_dice = 0
        total_train_loss = 0
        total_val_dice = 0
        total_val_loss = 0

if __name__ == "__main__":

    torch.multiprocessing.freeze_support()
    # parse the cli arguments here
    parser = argparse.ArgumentParser(description = "Training Loop of GANDLF")
    parser.add_argument('-train_loader_pickle', type=str, help = 'Train loader pickle', required=True)
    parser.add_argument('-val_loader_pickle', type=str, help = 'Validation loader pickle', required=True)
    parser.add_argument('-parameter_pickle', type=str, help = 'Parameters pickle', required=True)
    parser.add_argument('-headers_pickle', type=str, help = 'Header pickle', required=True)
    parser.add_argument('-outputDir', type=str, help = 'Output directory', required=True)
    parser.add_argument('-device', type=str, help = 'Device to train on', required=True)
    
    args = parser.parse_args()

    # # write parameters to pickle - this should not change for the different folds, so keeping is independent
    headers = pickle.load(open(args.headers_pickle,"rb"))
    parameters = pickle.load(open(args.parameter_pickle,"rb"))
    trainingDataFromPickle = pd.read_pickle(args.train_loader_pickle)
    validataionDataFromPickle = pd.read_pickle(args.val_loader_pickle)

    trainingLoop(trainingDataFromPickle=trainingDataFromPickle, 
                 validataionDataFromPickle=validataionDataFromPickle, 
                 headers = headers,  
                 parameters=parameters,
                 outputDir=args.outputDir,
                 device=args.device,)
