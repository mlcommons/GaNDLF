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
from GANDLF.losses import *
from GANDLF.utils import *
from .parameterParsing import *

def trainingLoop(trainingDataFromPickle, validationDataFromPickle, headers, device, parameters, outputDir, testingDataFromPickle = None):
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
    preprocessing = parameters['data_preprocessing']
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
    scaling_factor = parameters['scaling_factor']
  
    if len(psize) == 2:
        psize.append(1) # ensuring same size during torchio processing

    trainingDataForTorch = ImagesFromDataFrame(trainingDataFromPickle, psize, headers, q_max_length, q_samples_per_volume,
                                               q_num_workers, q_verbose, sampler = parameters['patch_sampler'], train=True, augmentations=augmentations, preprocessing = preprocessing)
    validationDataForTorch = ImagesFromDataFrame(validationDataFromPickle, psize, headers, q_max_length, q_samples_per_volume,
                                               q_num_workers, q_verbose, sampler = parameters['patch_sampler'], train=False, augmentations=augmentations, preprocessing = preprocessing) # may or may not need to add augmentations here
    if testingDataFromPickle is None:
        print('No testing data is defined, using validation data for those metrics')
        testingDataFromPickle = validationDataFromPickle
    inferenceDataForTorch = ImagesFromDataFrame(testingDataFromPickle, psize, headers, q_max_length, q_samples_per_volume,
                                            q_num_workers, q_verbose, sampler = parameters['patch_sampler'], train=False, augmentations=augmentations, preprocessing = preprocessing)
    
    
    train_loader = DataLoader(trainingDataForTorch, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validationDataForTorch, batch_size=1)
    inference_loader = DataLoader(inferenceDataForTorch,batch_size=1)
    
    # Defining our model here according to parameters mentioned in the configuration file
    model = get_model(which_model, parameters['dimension'], n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'], psize = psize)

    
    is_regression = False
    is_classification = False

    # check if regression/classification has been requested
    if len(headers['predictionHeaders']) > 0:
        if model.final_convolution_layer is None:
            is_regression = True
        else:
            is_classification = True

    # sanity check
    if n_channels == 0:
        sys.exit('The number of input channels cannot be zero, please check training CSV')

    # setting optimizer
    optimizer = get_optimizer(opt, model.parameters(), learning_rate) 
        
    # setting the loss function
    loss_fn, MSE_requested = get_loss(loss_function)

    # training_start_time = time.asctime()
    # startstamp = time.time()
    if not(os.environ.get('HOSTNAME') is None):
        print("\nHostname     :" + str(os.environ.get('HOSTNAME')))
        sys.stdout.flush()

    # resume if compatible model was found
    if os.path.exists(os.path.join(outputDir,str(which_model) + "_best.pth.tar")):
        checkpoint = torch.load(os.path.join(outputDir,str(which_model) + "_best.pth.tar"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model checkpoint found. Loading checkpoint from: ",os.path.join(outputDir,str(which_model) + "_best.pth.tar"))

    print("Samples - Train: %d Val: %d Test: %d"%(len(train_loader.dataset),len(val_loader.dataset),len(inference_loader.dataset)))
    sys.stdout.flush()

    model, amp, device = send_model_to_device(model, amp, device, optimizer=optimizer)
    print('Using device:', device)        
    sys.stdout.flush()

    # Checking for the learning rate scheduler
    scheduler_lr = get_scheduler(scheduler, optimizer, batch_size, len(train_loader.dataset), learning_rate)

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
    log_train.write("Epoch,Train_Loss,Train_Dice,Val_Loss,Val_Dice,Testing_Loss,Testing_Dice\n")
    log_train.close()

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
    
    # For regression dice penalty need not be taken account
    # For classification this should be calculated on the basis of predicted labels and mask
    if not is_regression:
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
     
    # Getting the channels for training and removing all the non numeric entries from the channels
    batch = next(iter(train_loader))
    all_keys = list(batch.keys())
    channel_keys = []
    value_keys = []

    for item in all_keys:
        if item.isnumeric():
            channel_keys.append(item)
        elif 'value' in item:
            value_keys.append(item)

    # automatic mixed precision - https://pytorch.org/docs/stable/amp.html
    if amp:
        scaler = torch.cuda.amp.GradScaler() 

    ################ TRAINING THE MODEL##############
    for ep in range(num_epochs):
        start = time.time()
        print("\n")
        print("Ep# %03d | LR: %s | Start: %s "%(ep, str(optimizer.param_groups[0]['lr']), str(datetime.datetime.now())))
        model.train()
        for batch_idx, (subject) in enumerate(train_loader):
            # uncomment line to debug memory issues
            # # print('=== Memory (allocated; cached) : ', round(torch.cuda.memory_allocated(int(dev))/1024**3, 1), '; ', round(torch.cuda.memory_reserved(int(dev))/1024**3, 1))
            # Load the subject and its ground truth
            # read and concat the images
            image = torch.cat([subject[key][torchio.DATA] for key in channel_keys], dim=1) # concatenate channels 
            
            # if regression, concatenate values to predict
            if is_regression:
                valuesToPredict = torch.cat([subject[key] for key in value_keys], dim=0)
                valuesToPredict = torch.reshape(subject[value_keys[0]], (batch_size,1))
                valuesToPredict = valuesToPredict*scaling_factor
                if device.type != 'cpu':
                    valuesToPredict = valuesToPredict.to(device)
            
            # read the mask
            if subject['label'] == ['NA']:
                mask_present = False
            else:
                mask_present = True
                mask = subject['label'][torchio.DATA] # get the label image

            ## special case for 2D            
            if image.shape[-1] == 1:
                model_2d = True
                image = torch.squeeze(image, -1)
                if mask_present:
                    mask = torch.squeeze(mask, -1)
            else:
                model_2d = False
            # Why are we doing this? Please check again
            #mask = one_hot(mask.cpu().float().numpy(), class_list)
            if mask_present:
                one_hot_mask = one_hot(mask, class_list)
            # one_hot_mask = one_hot_mask.unsqueeze(0)
            #mask = torch.from_numpy(mask)
            # Loading images into the GPU and ignoring the affine
            image = image.float().to(device)
            if mask_present:
                one_hot_mask = one_hot_mask.to(device)

            # Making sure that the optimizer has been reset
            optimizer.zero_grad()
            # Forward Propagation to get the output from the models
            # TODO: Not recommended? (https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/6)will try without
            # might help solve OOM
            # torch.cuda.empty_cache()
            # Casts operations to mixed precision
            output = model(image)
            if is_regression:
                #print("Output:", output) #U
                #print("Values to predict:", valuesToPredict)  #U
                output = output.clone().type(dtype=torch.float) #U
                valuesToPredict = valuesToPredict.clone().type(dtype=torch.float) #U

                #loss = MSE(output, valuesToPredict) 
                loss = torch.nn.MSELoss()(output, valuesToPredict)
                print("loss:", loss)
                loss.backward()
                optimizer.step()
            else:
                if model_2d: # for 2D, add a dimension so that loss can be computed without modifications
                    one_hot_mask = one_hot_mask.unsqueeze(-1)
                    output = output.unsqueeze(-1)
            
            if not is_regression:
                if amp:
                    with torch.cuda.amp.autocast(): 
                    # Computing the loss
                        if MSE_requested:
                            loss = loss_fn(output.double(), one_hot_mask.double(), n_classList, reduction = loss_function['mse']['reduction'])
                        else:
                            loss = loss_fn(output.double(), one_hot_mask.double(), n_classList, dice_penalty_dict)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                else:
                    # Computing the loss
                    if MSE_requested:
                        loss = loss_fn(output.double(), one_hot_mask.double(), n_classList, reduction = loss_function['mse']['reduction'])
                    else:
                        loss = loss_fn(output.double(), one_hot_mask.double(), n_classList, dice_penalty_dict)
                    loss.backward()
                    optimizer.step()

            
            
                           
            #Pushing the dice to the cpu and only taking its value
            curr_loss = loss.cpu().data.item()
            #train_loss_list.append(loss.cpu().data.item())
            total_train_loss += curr_loss

            if not is_regression:
                #Computing the dice score  # Can be changed for multi-class outputs later.
                curr_dice = MCD(output.double(), one_hot_mask.double(), n_classList).cpu().data.item() # https://discuss.pytorch.org/t/cuda-memory-leakage/33970/3
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
            #print(curr_dice)

        #average_train_dice = total_train_dice/len(train_loader.dataset) #U
        average_train_loss = total_train_loss/len(train_loader.dataset)
                               
        # if average_train_dice > best_train_dice:
        #     best_train_idx = ep
        #     best_train_dice = average_train_dice
        #     torch.save({"epoch": best_train_idx,
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        #     "best_train_dice": best_train_dice }, os.path.join(outputDir, which_model + "_best_train.pth.tar"))

        # print("   Train DCE: ", format(average_train_dice,'.10f'), " | Best Train DCE: ", format(best_train_dice,'.10f'), " | Avg Train Loss: ", format(average_train_loss,'.10f'), " | Best Train Ep ", format(best_train_idx,'.1f'))

        # # Now we enter the evaluation/validation part of the epoch      
        # # validation data scores
        # average_val_dice, average_val_loss = get_metrics_save_mask(model, device, val_loader, psize, channel_keys, class_list, loss_fn)

        # # testing data scores
        # average_test_dice, average_test_loss = get_metrics_save_mask(model, device, inference_loader, psize, channel_keys, class_list, loss_fn) 
        
        # # stats for current validation data
        # if average_val_dice > best_val_dice:
        #     best_val_idx = ep
        #     best_val_dice = average_val_dice
        #     best_test_val_dice = average_val_dice
        #     # We can add more stuff to be saved if we need anything more
        #     torch.save({"epoch": best_val_idx,
        #                 "model_state_dict": model.state_dict(),
        #                 "optimizer_state_dict": optimizer.state_dict(),
        #                 "best_val_dice": best_val_dice }, os.path.join(outputDir, which_model + "_best_val.pth.tar"))
        # else:
        #     patience_count = patience_count + 1 
        # print("     Val DCE: ", format(average_val_dice,'.10f'), " | Best Val   DCE: ", format(best_val_dice,'.10f'), " | Avg Train Loss: ", format(average_val_loss,'.10f'), " | Best Val   Ep ", format(best_val_idx,'.1f'))

        # # stats for current testing data
        # if average_test_dice > best_test_dice:
        #     best_test_idx = ep
        #     best_test_dice = average_test_dice
        #     # We can add more stuff to be saved if we need anything more
        #     torch.save({"epoch": best_test_idx,
        #                 "model_state_dict": model.state_dict(),
        #                 "optimizer_state_dict": optimizer.state_dict(),
        #                 "best_test_dice": best_test_dice }, os.path.join(outputDir, which_model + "_best_test.pth.tar"))
        # print("    Test DCE: ", format(average_test_dice,'.10f'), " | Best Test  DCE: ", format(best_test_dice,'.10f'), " | Avg Train Loss: ", format(average_test_loss,'.10f'), " | Best Test  Ep ", format(best_test_idx,'.1f'))

        # Updating the learning rate according to some conditions - reduce lr on plateau needs out loss to be monitored and schedules the LR accordingly. Others change irrespective of loss.
        if not scheduler == "triangular":
            if scheduler == "reduce-on-plateau":
                scheduler_lr.step(average_val_loss)
            else:
                scheduler_lr.step()

        # Saving the current model
        # torch.save({"epoch": ep,
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "val_dice": average_val_dice }, os.path.join(outputDir, which_model + "_latest.pth.tar"))

        stop = time.time()     
        print("Time for epoch: ",(stop - start)/60," mins")        

        # Checking if patience is crossed
        if patience_count > patience:
            print("Performance Metric has not improved for %d epochs, exiting training loop"%(patience))
            break
        
        sys.stdout.flush()
        log_train = open(log_train_file, "a")
        #log_train.write(str(ep) + "," + str(average_train_loss) + "," + str(average_train_dice) + "," + str(average_val_loss) + "," + str(average_val_dice) + "," + str(average_test_loss) + "," + str(average_test_dice) + "\n")
        log_train.close()
        total_train_dice = 0
        total_train_loss = 0

if __name__ == "__main__":

    torch.multiprocessing.freeze_support()
    # parse the cli arguments here
    parser = argparse.ArgumentParser(description = "Training Loop of GANDLF")
    parser.add_argument('-train_loader_pickle', type=str, help = 'Train loader pickle', required=True)
    parser.add_argument('-val_loader_pickle', type=str, help = 'Validation loader pickle', required=True)
    parser.add_argument('-testing_loader_pickle', type=str, help = 'Testing loader pickle', required=True)
    parser.add_argument('-parameter_pickle', type=str, help = 'Parameters pickle', required=True)
    parser.add_argument('-headers_pickle', type=str, help = 'Header pickle', required=True)
    parser.add_argument('-outputDir', type=str, help = 'Output directory', required=True)
    parser.add_argument('-device', type=str, help = 'Device to train on', required=True)
    
    args = parser.parse_args()

    # # write parameters to pickle - this should not change for the different folds, so keeping is independent
    headers = pickle.load(open(args.headers_pickle,"rb"))
    parameters = pickle.load(open(args.parameter_pickle,"rb"))
    trainingDataFromPickle = pd.read_pickle(args.train_loader_pickle)
    validationDataFromPickle = pd.read_pickle(args.val_loader_pickle)
    testingData_str = args.testing_loader_pickle
    if testingData_str == 'None':
        testingDataFromPickle = None
    else:
        testingDataFromPickle = pd.read_pickle(testingData_str)

    trainingLoop(trainingDataFromPickle=trainingDataFromPickle, 
                 validationDataFromPickle=validationDataFromPickle, 
                 headers = headers,  
                 parameters=parameters,
                 outputDir=args.outputDir,
                 device=args.device,
                 testingDataFromPickle=testingDataFromPickle,)
