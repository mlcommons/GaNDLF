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
from copy import deepcopy
import time, math
import sys
import pickle
from pathlib import Path
import argparse
import datetime
import SimpleITK as sitk
from GANDLF.utils import *
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.schd import *
from GANDLF.losses import *
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
    opt = parameters['opt']
    loss_function = parameters['loss_function']
    scheduler = parameters['scheduler']
    batch_size = parameters['batch_size']
    learning_rate = parameters['learning_rate']
    num_epochs = parameters['num_epochs']
    amp = parameters['model']['amp']
    patience = parameters['patience']
    use_weights = parameters['weighted_loss']
    in_memory = parameters['in_memory']

    ## model configuration
    which_model = parameters['model']['architecture']
    dimension = parameters['model']['dimension']
    base_filters = parameters['model']['base_filters']
    class_list = parameters['model']['class_list']
    scaling_factor = parameters['scaling_factor']
    n_classList = len(class_list)

    if not('n_channels' in parameters['model']):
        n_channels = len(headers['channelHeaders'])
    else:
        n_channels = parameters['model']['n_channels']

    # Defining our model here according to parameters mentioned in the configuration file
    print("Number of dims     : ", parameters['model']['dimension'])
    if 'n_channels' in parameters['model']:
        print("Number of channels : ", parameters['model']['n_channels'])
    
    if len(headers['predictionHeaders']) > 0: # for regressin/classification
        n_classList = len(headers['predictionHeaders']) 
    print("Number of classes  : ", n_classList)
    model = get_model(which_model, dimension, n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'], psize = psize, batch_size = batch_size, batch_norm = parameters['model']['batch_norm'])

    # initialize problem type    
    is_regression, is_classification, is_segmentation = find_problem_type(headers, model.final_convolution_layer)

    if is_regression or is_classification:
        n_classList = len(headers['predictionHeaders']) # ensure the output class list is correctly populated
  
    trainingDataForTorch = ImagesFromDataFrame(trainingDataFromPickle, psize, headers, q_max_length, q_samples_per_volume,
                                               q_num_workers, q_verbose, sampler = parameters['patch_sampler'], train=True, augmentations=augmentations, preprocessing = preprocessing, in_memory=in_memory)
    validationDataForTorch = ImagesFromDataFrame(validationDataFromPickle, psize, headers, q_max_length, q_samples_per_volume,
                                               q_num_workers, q_verbose, sampler = parameters['patch_sampler'], train=False, augmentations=augmentations, preprocessing = preprocessing, in_memory=in_memory) # may or may not need to add augmentations here
    testingDataDefined = True
    if testingDataFromPickle is None:
        print('No testing data is defined, using validation data for those metrics', flush=True)
        testingDataFromPickle = validationDataFromPickle
        testingDataDefined = False
    inferenceDataForTorch = ImagesFromDataFrame(testingDataFromPickle, psize, headers, q_max_length, q_samples_per_volume,
                                            q_num_workers, q_verbose, sampler = parameters['patch_sampler'], train=False, augmentations=augmentations, preprocessing = preprocessing)
    
    train_loader = DataLoader(trainingDataForTorch, batch_size=batch_size, shuffle=True, pin_memory=in_memory)
    val_loader = DataLoader(validationDataForTorch, batch_size=1)
    inference_loader = DataLoader(inferenceDataForTorch,batch_size=1)
    
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
        print("\nHostname     :" + str(os.environ.get('HOSTNAME')), flush=True)

    # resume if compatible model was found
    if os.path.exists(os.path.join(outputDir,str(which_model) + "_best.pth.tar")):
        checkpoint = torch.load(os.path.join(outputDir,str(which_model) + "_best.pth.tar"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model checkpoint found. Loading checkpoint from: ",os.path.join(outputDir,str(which_model) + "_best.pth.tar"))

    print("Samples - Train: %d Val: %d Test: %d"%(len(train_loader.dataset),len(val_loader.dataset),len(inference_loader.dataset)), flush=True)

    model, amp, device = send_model_to_device(model, amp, device, optimizer=optimizer)
    print('Using device:', device, flush=True)

    # Checking for the learning rate scheduler
    scheduler_lr = get_scheduler(scheduler, optimizer, batch_size, len(train_loader.dataset), learning_rate)

    ############## STORING THE HISTORY OF THE LOSSES #################
    best_val_dice = best_train_dice = best_test_dice = -1
    best_val_loss = best_train_loss = best_test_loss =  1000000
    total_train_loss = total_train_dice = 0
    patience_count = 0    
    best_train_idx = best_val_idx = best_test_idx = 0

    # Creating a CSV to log training loop and writing the initial columns
    log_train_file = os.path.join(outputDir,"trainingScores_log.csv")
    log_train = open(log_train_file,"w")
    log_train.write("Epoch,Train_Loss,Train_Dice,Val_Loss,Val_Dice,Testing_Loss,Testing_Dice\n")
    log_train.close()

    if use_weights:
        print('Calculating penalty weights', flush=True)
        dice_weights_dict = {} # average for "weighted averaging"
        dice_penalty_dict = {} # penalty for misclassification
        for i in range(0, n_classList):
            dice_weights_dict[i] = 0
            dice_penalty_dict[i] = 0
        # define a seaparate data loader for penalty calculations
        penaltyData = ImagesFromDataFrame(trainingDataFromPickle, psize, headers, q_max_length, q_samples_per_volume, q_num_workers, q_verbose, sampler = parameters['patch_sampler'], train=False, augmentations=None, preprocessing=None) 
        penalty_loader = DataLoader(penaltyData, batch_size=1)
        
        # get the weights for use for dice loss
        total_nonZeroVoxels = 0
        
        # For regression dice penalty need not be taken account
        # For classification this should be calculated on the basis of predicted labels and mask
        if is_segmentation:
            for batch_idx, (subject) in enumerate(penalty_loader): # iterate through full training data
                # accumulate dice weights for each label
                mask = subject['label'][torchio.DATA]
                one_hot_mask = one_hot(mask, class_list)
                for i in range(0, n_classList):
                    currentNumber = torch.nonzero(one_hot_mask[:,i,:,:,:], as_tuple=False).size(0)
                    dice_weights_dict[i] = dice_weights_dict[i] + currentNumber # class-specific non-zero voxels
                    total_nonZeroVoxels = total_nonZeroVoxels + currentNumber # total number of non-zero voxels to be considered
                
            if total_nonZeroVoxels == 0:
                raise RuntimeError('Trying to train on data where every label mask is background class only.')

            # dice_weights_dict_temp = deepcopy(dice_weights_dict)
            dice_weights_dict = {k: (v / total_nonZeroVoxels) for k, v in dice_weights_dict.items()} # divide each dice value by total nonzero
            dice_penalty_dict = deepcopy(dice_weights_dict) # deep copy so that both values are preserved
            dice_penalty_dict = {k: 1 - v for k, v in dice_weights_dict.items()} # subtract from 1 for penalty
            total = sum(dice_penalty_dict.values())
            dice_penalty_dict = {k: v / total for k, v in dice_penalty_dict.items()} # normalize penalty to ensure sum of 1
            # dice_penalty_dict = get_class_imbalance_weights(trainingDataFromPickle, parameters, headers, is_regression, class_list) # this doesn't work because ImagesFromDataFrame gets import twice, causing a "'module' object is not callable" error
    else:
        dice_penalty_dict = None
        # initialize without considering background
        
    # Getting the channels for training and removing all the non numeric entries from the channels

    batch = next(iter(val_loader)) # using train_loader makes this slower as train loader contains full augmentations
    all_keys = list(batch.keys())
    channel_keys = []
    value_keys = []
    print("Channel Keys : ", all_keys)
    for item in all_keys:
        if item.isnumeric():
            channel_keys.append(item)
        elif 'value' in item:
            value_keys.append(item)

    # automatic mixed precision - https://pytorch.org/docs/stable/amp.html
    if amp:
        print('Using automatic mixed precision', flush=True)
        scaler = torch.cuda.amp.GradScaler() 

    ################ TRAINING THE MODEL##############
    for ep in range(num_epochs):
        start = time.time()
        print("\nEp# %03d | LR: %s | Start: %s "%(ep, str(optimizer.param_groups[0]['lr']), str(datetime.datetime.now())), flush=True)
        samples_for_train = 0
        model.train()
        for batch_idx, (subject) in enumerate(train_loader):
            # uncomment line to debug memory issues
            # # print('=== Memory (allocated; cached) : ', round(torch.cuda.memory_allocated(int(dev))/1024**3, 1), '; ', round(torch.cuda.memory_reserved(int(dev))/1024**3, 1))
            # Load the subject and its ground truth
            # read and concat the images
            image = torch.cat([subject[key][torchio.DATA] for key in channel_keys], dim=1) # concatenate channels
            
            # if regression, concatenate values to predict
            if is_regression or is_classification:
                valuesToPredict = torch.cat([subject[key] for key in value_keys], dim=0)
                valuesToPredict = torch.reshape(subject[value_keys[0]], (batch_size,1))
                valuesToPredict = valuesToPredict*scaling_factor
                if device.type != 'cpu':
                    valuesToPredict = valuesToPredict.to(device)
            
            # read the mask
            first = next(iter(subject['label']))
            if first == 'NA':
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
                #temp = reverse_one_hot(one_hot_mask, class_list)
            # one_hot_mask = one_hot_mask.unsqueeze(0)
            #mask = torch.from_numpy(mask)
            # Loading images into the GPU and ignoring the affine
            if device.type != 'cpu':
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
            if is_regression or is_classification:
                #print("Output:", output) #U
                #print("Values to predict:", valuesToPredict)  #U
                output = output.clone().type(dtype=torch.float) #U
                valuesToPredict = valuesToPredict.clone().type(dtype=torch.float) #U

                #loss = MSE(output, valuesToPredict) 
                loss = torch.nn.MSELoss()(output, valuesToPredict)
                curr_loss = loss.cpu().data.item()
                if amp:
                    with torch.cuda.amp.autocast(): 
                        if not math.isnan(curr_loss): # if loss is nan, dont backprop and dont step optimizer
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                else:
                    if not math.isnan(curr_loss): # if loss is nan, dont backprop and dont step optimizer
                        loss.backward()
                        optimizer.step()
            
            else: # segmentation
                if model_2d: # for 2D, add a dimension so that loss can be computed without modifications
                    one_hot_mask = one_hot_mask.unsqueeze(-1)
                    output = output.unsqueeze(-1)
                # Computing the loss
                if MSE_requested:
                    loss = loss_fn(output.double(), one_hot_mask.double(), n_classList, reduction = loss_function['mse']['reduction'])
                else:
                    loss = loss_fn(output.double(), one_hot_mask.double(), n_classList, dice_penalty_dict)
                curr_loss = loss.cpu().data.item()
                if amp:
                    with torch.cuda.amp.autocast(): 
                        if not math.isnan(curr_loss): # if loss is nan, dont backprop and dont step optimizer
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                else:
                    if not math.isnan(curr_loss): # if loss is nan, dont backprop and dont step optimizer
                        loss.backward()
                        optimizer.step()

            #Pushing the dice to the cpu and only taking its value
            # print('=== curr_loss: ', curr_loss)
            # curr_loss = loss.cpu().data.item()
            #train_loss_list.append(loss.cpu().data.item())
            ## debugging new loss
            # temp_loss = MCD_loss_new(output.double(), one_hot_mask.double(), n_classList, dice_penalty_dict).cpu().data.item()
            # print('curr_loss:', curr_loss)
            # print('temp_loss:', temp_loss)
            ## debugging new loss

            if not math.isnan(curr_loss):
                total_train_loss += curr_loss
            samples_for_train += 1

            if is_segmentation:
                #Computing the dice score  # Can be changed for multi-class outputs later.
                curr_dice = MCD(output.double(), one_hot_mask.double(), n_classList).cpu().data.item() # https://discuss.pytorch.org/t/cuda-memory-leakage/33970/3
                #print(curr_dice)
                # print('=== curr_dice: ', curr_dice)
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

        if is_segmentation:
            average_train_dice = total_train_dice/samples_for_train #len(train_loader.dataset) 
        else:
            average_train_dice = 1

        average_train_loss = total_train_loss/samples_for_train #len(train_loader.dataset)
        # print('total_train_loss:', total_train_loss)
        # print('average_train_loss:', average_train_loss)
        # print('average_train_loss_old:', total_train_loss/len(train_loader.dataset))

        # initialize some bool variables to control model saving
        save_condition_train = False
        save_condition_val = False
        save_condition_test = False

        # Now we enter the evaluation/validation part of the epoch      
        # validation data scores
        average_val_dice, average_val_loss = get_metrics_save_mask(model, device, val_loader, psize, channel_keys, value_keys, class_list, loss_fn, is_segmentation, scaling_factor, save_mask=parameters['save_masks'], outputDir = os.path.join(outputDir, 'validationOutput'), ignore_label_validation=parameters['model']['ignore_label_validation'], num_patches=parameters['q_samples_per_volume'])

        # testing data scores
        average_test_dice, average_test_loss = get_metrics_save_mask(model, device, inference_loader, psize, channel_keys, value_keys, class_list, loss_fn, is_segmentation, scaling_factor, save_mask=parameters['save_masks'] & testingDataDefined, outputDir = os.path.join(outputDir, 'testingoutput'), ignore_label_validation=parameters['model']['ignore_label_validation'], num_patches=parameters['q_samples_per_volume'])
    
        # regression or classification, use the loss to drive the model saving
        if is_segmentation:
            save_condition_train = average_train_dice > best_train_dice
            if save_condition_train:
                best_train_dice = average_train_dice
            save_condition_val = average_val_dice > best_val_dice
            if save_condition_val:
                best_val_dice = average_val_dice
                patience_count = 0
            else: # patience is calculated on validation
                patience_count = patience_count + 1 
            save_condition_test = average_test_dice > best_test_dice
            if save_condition_test:
                best_test_dice = average_test_dice
        else: 
            save_condition_train = average_train_loss < best_train_loss
            if save_condition_train:
                best_train_loss = average_train_loss
            save_condition_val = average_val_loss < best_val_loss
            if save_condition_val:
                best_val_loss = average_val_loss
            else: # patience is calculated on validation
                patience_count = patience_count + 1 
            save_condition_test = average_test_loss < best_test_loss
            if save_condition_test:
                best_test_loss = average_test_loss

        if save_condition_train:
            best_train_idx = ep
            torch.save({"epoch": best_train_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_train_dice": best_train_dice,
            "best_train_loss": best_train_loss }, os.path.join(outputDir, which_model + "_best_train.pth.tar"))
            
        print("   Train Dice: ", format(average_train_dice,'.10f'), " | Best Train Dice: ", format(best_train_dice,'.10f'), " | Avg Train Loss: ", format(average_train_loss,'.10f'), " | Best Train Ep ", format(best_train_idx,'.0f'), flush=True)

        if save_condition_val:
            best_val_idx = ep
            torch.save({"epoch": best_val_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": best_val_dice,
            "best_val_loss": best_val_loss }, os.path.join(outputDir, which_model + "_best_val.pth.tar"))
        
        print("     Val Dice: ", format(average_val_dice,'.10f'), " | Best Val   Dice: ", format(best_val_dice,'.10f'), " | Avg Val   Loss: ", format(average_val_loss,'.10f'), " | Best Val   Ep ", format(best_val_idx,'.0f'), flush=True)

        if save_condition_test:
            best_test_idx = ep
            torch.save({"epoch": best_test_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_test_dice": best_test_dice,
            "best_test_loss": best_test_loss }, os.path.join(outputDir, which_model + "_best_test.pth.tar"))

        print("    Test Dice: ", format(average_test_dice,'.10f'), " | Best Test  Dice: ", format(best_test_dice,'.10f'), " | Avg Test  Loss: ", format(average_test_loss,'.10f'), " | Best Test  Ep ", format(best_test_idx,'.0f'), flush=True)

        # Updating the learning rate according to some conditions - reduce lr on plateau needs out loss to be monitored and schedules the LR accordingly. Others change irrespective of loss.
        
        if not scheduler == "triangular":
            if scheduler == "reduce-on-plateau":
                scheduler_lr.step(average_val_loss)
            else:
                scheduler_lr.step()

        #Saving the current model
        torch.save({"epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": average_val_dice, "val_loss": average_val_loss }, os.path.join(outputDir, which_model + "_latest.pth.tar"))

        stop = time.time()     
        print("Time for epoch: ",(stop - start)/60," mins", flush=True)        

        # Checking if patience is crossed
        if patience_count > patience:
            print("Performance Metric has not improved for %d epochs, exiting training loop"%(patience), flush=True)
            break
        
        log_train = open(log_train_file, "a")
        log_train.write(str(ep) + "," + str(average_train_loss) + "," + str(average_train_dice) + "," + str(average_val_loss) + "," + str(average_val_dice) + "," + str(average_test_loss) + "," + str(average_test_dice) + "\n")
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
