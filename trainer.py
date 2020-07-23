
from __future__ import print_function, division
import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235

from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
import data
from data.ImagesFromDataFrame import ImagesFromDataFrame
from data_val import TumorSegmentationDataset_val
from schd import *
from models.fcn import fcn
from models.unet import unet
from models.resunet import resunet
from models.uinc import uinc
import gc
from torchsummary import summary
import nibabel as nib
from losses import *
import sys
import ast 
import datetime
from pathlib import Path
from sklearn.model_selection import KFold
import pickle
import pkg_resources
from shutil import copyfile
import torchio

parser = argparse.ArgumentParser(description = "3D Image Semantic Segmentation using Deep Learning")
parser.add_argument('-m', '--model', type=str, help = 'model configuration file', required=True)
parser.add_argument('-d', '--data', type=str, help = 'data csv file that is used for training or testing', required=True)
parser.add_argument('-o', '--output', type=str, help = 'output directory to save intermediate files and model weights', required=True)
parser.add_argument('-tr', '--train', default=1, type=int, help = '1 means training and 0 means testing; for 0, there needs to be a compatible model saved in \'-md\'', required=False)
parser.add_argument('-md', '--modelDir', type=str, help = 'The pre-trained model directory that is used for testing', required=False)
parser.add_argument('-dv', '--device', default=0, type=int, help = 'choose device', required=True) # todo: how to handle cpu training? would passing '-1' be considered cpu?
parser.add_argument('-s', '--sge', default=0, type=int, help = 'Whether the training is running on SGE for parallel fold training across nodes', required=False) # todo: how to handle cpu training? would passing '-1' be considered cpu?
parser.add_argument('-sm', '--sgeMem', default=64, type=int, help = 'The amount of memory requested per training job for SGE; used only when \'-s 1\' is passed', required=False) # todo: how to handle cpu training? would passing '-1' be considered cpu?
parser.add_argument('-v', '--version', action='version', version=pkg_resources.require('deep-seg')[0].version, help="Show program's version number and exit.")
                            
args = parser.parse_args()

file_trainingData_full = args.data
model_parameters = args.model
dev = args.device
model_path = args.output
mode = args.train
if dev>=0:
    dev = 'cuda'
if dev==-1:
    dev = 'cpu'
sge_run = args.sge
if sge_run == 0:
    sge_run = False
else:
    sge_run = True
sge_memory = args.sgeMem

if mode == 0:
    pretrainedModelPath = args.modelDir

# safe directory creation
Path(model_path).mkdir(parents=True, exist_ok=True)

df_model = pd.read_csv(model_parameters, sep=' = ', names=['param_name', 'param_value'],
                         comment='#', skip_blank_lines=True,
                         engine='python').fillna(' ')

#Read the parameters as a dictionary so that we can access everything by the name and so when we add some extra parameters we dont have to worry about the indexing
params = {}
for j in range(df_model.shape[0]):
    params[df_model.iloc[j, 0]] = df_model.iloc[j, 1]

# Extrating the training parameters from the dictionary
num_epochs = int(params['num_epochs'])
batch_size = int(params['batch_size'])
learning_rate = int(params['learning_rate'])
which_loss = str(params['loss_function'])
opt = str(params['opt'])
save_best = int(params['save_best'])
augmentations = ast.literal_eval(str(params['data_augmentation']))

# Extracting the model parameters from the dictionary
n_classes = int(params['numberOfOutputClasses'])
base_filters = int(params['base_filters'])
n_channels = int(params['numberOfInputChannels'])
# model_path = str(params['folderForOutput'])
which_model = str(params['modelName'])
kfolds = int(params['kcross_validation'])
psize = params['patch_size']
psize = ast.literal_eval(psize) 
psize = np.array(psize)

# Defining our model here according to parameters mentioned in the configuration file : 
if which_model == 'resunet':
    model = resunet(n_channels,n_classes,base_filters)
elif which_model == 'unet':
    model = unet(n_channels,n_classes,base_filters)
elif which_model == 'fcn':
    model = fcn(n_channels,n_classes,base_filters)
elif which_model == 'uinc':
    model = uinc(n_channels,n_classes,base_filters)
else:
    print('WARNING: Could not find the requested model \'' + which_model + '\' in the impementation, using ResUNet, instead', file = sys.stderr)
    which_model = 'resunet'
    model = resunet(n_channels,n_classes,base_filters)

# setting optimizer
if opt == 'sgd':
    optimizer = optim.SGD(model.parameters(),
                               lr= learning_rate,
                               momentum = 0.9)
elif opt == 'adam':    
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9,0.999), weight_decay = 0.00005)
else:
    print('WARNING: Could not find the requested optimizer \'' + opt + '\' in the impementation, using sgd, instead', file = sys.stderr)
    opt = 'sgd'
    optimizer = optim.SGD(model.parameters(),
                               lr= learning_rate,
                               momentum = 0.9)
# setting the loss function
if which_loss == 'dc':
    loss_fn  = MCD_loss
elif which_loss == 'dcce':
    loss_fn  = DCCE
elif which_loss == 'ce':
    loss_fn = CE
elif which_loss == 'mse':
    loss_fn = MCD_MSE_loss
else:
    print('WARNING: Could not find the requested loss function \'' + which_loss + '\' in the impementation, using dc, instead', file = sys.stderr)
    which_loss = 'dc'
    loss_fn  = MCD_loss

## read training dataset into data frame
trainingData_full = pd.read_csv(file_trainingData_full)
# shuffle the data - this is a useful level of randomization for the training process
trainingData_full=trainingData_full.sample(frac=1).reset_index(drop=True)

# find actual header locations for input channel and label
# the user might put the label first and the channels afterwards 
# or might do it completely randomly
channelHeaders = []
for col in trainingData_full.columns: 
    # add appropriate headers to read here, as needed
    if ('Channel' in col) or ('Modality' in col) or ('Image' in col):
        channelHeaders.append(trainingData_full.columns.get_loc(col))
    elif ('Label' in col) or ('Mask' in col) or ('Segmentation' in col):
        labelHeader = trainingData_full.columns.get_loc(col)

# get the indeces for kfold splitting
training_indeces_full = list(trainingData_full.index.values)

# check for single fold training
singleFoldTraining = False
if kfolds < 0: # if the user wants a single fold training
    kfolds = abs(kfolds)
    singleFoldTraining = True

kf = KFold(n_splits=kfolds) # initialize the kfold structure

currentFold = 0

# write parameters to pickle - this should not change for the different folds, so keeping is independent
paramtersPickle = os.path.join(model_path,'params.pkl')
with open(paramtersPickle, 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# start the kFold train
for train_index, test_index in kf.split(training_indeces_full):

    # the output of the current fold is only needed if multi-fold training is happening
    if singleFoldTraining:
        currentOutputFolder = model_path
    else:
        currentOutputFolder = os.path.join(model_path, str(currentFold))
        Path(currentOutputFolder).mkdir(parents=True, exist_ok=True)

    trainingData = trainingData_full.iloc[train_index]
    validationData = trainingData_full.iloc[test_index]

    # save the current model configuration as a sanity check
    copyfile(model_parameters, os.path.join(currentOutputFolder,'model.cfg'))

    # pickle the data
    currentTrainingDataPickle = os.path.join(currentOutputFolder, 'train.pkl')
    currentValidataionDataPickle = os.path.join(currentOutputFolder, 'validation.pkl')
    trainingData.to_pickle(currentTrainingDataPickle)
    validationData.to_pickle(currentValidataionDataPickle)


    ### inside the training function
    ### for efficient processing, this can be passed off to sge as independant processes
    # trainingDataFromPickle = pd.read_pickle('/path/to/train.pkl')
    # validataionDataFromPickle = pd.read_pickle('/path/to/validation.pkl')
    # paramsPickle = pd.read_pickle('/path/to/validation.pkl')
    # with open('/path/to/params.pkl', 'rb') as handle:
    #     params = pickle.load(handle)

    trainingDataForTorch = ImagesFromDataFrame(trainingData, psize, channelHeaders, labelHeader, augmentations)
    validationDataForTorch = ImagesFromDataFrame(validationData, psize, channelHeaders, labelHeader, augmentations) # may or may not need to add augmentations here

    # construct the data queue using pre-defined information
    # all of this needs to come from the config file
    patch_size = 128 # Tuple of integers (d,h,w) to generate patches of size d×h×w. If a single number n is provided, d=h=w=n.
    queue_length = 100 # Maximum number of patches that can be stored in the queue. Using a large number means that the queue needs to be filled less often, but more CPU memory is needed to store the patches.
    samples_per_volume = 10 #  Number of patches to extract from each volume. A small number of patches ensures a large variability in the queue, but training will be slower.

    queue_training = torchio.Queue(
        trainingDataForTorch,
        queue_length,
        samples_per_volume,
        torchio.data.UniformSampler(patch_size),
    )
    
    queue_validation = torchio.Queue(
        validationDataForTorch,
        queue_length,
        samples_per_volume,
        torchio.data.UniformSampler(patch_size),
    )
    
    training_start_time = time.asctime()
    startstamp = time.time()
    print("\nHostname   :" + str(os.getenv("HOSTNAME")))
    sys.stdout.flush()

    # Setting up the train and validation loader
    train_loader = DataLoader(queue_training,batch_size=batch_size,shuffle=True,num_workers=1)
    val_loader = DataLoader(queue_validation, batch_size=1,shuffle=True,num_workers = 1)

    # get the channel keys
    batch = next(iter(train_loader))

    print("Training Data Samples: ", len(train_loader.dataset))
    sys.stdout.flush()
    device = torch.device(dev)
    print("Current Device : ", torch.cuda.current_device())
    print("Device Count on Machine : ", torch.cuda.device_count())
    print("Device Name : ", torch.cuda.get_device_name(device))
    print("Cuda Availibility : ", torch.cuda.is_available())
    print('Using device:', device)
    if device.type == 'cuda':
        print('Memory Usage:')
        print('  Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1),'GB')
        print('  Cached: ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')

    sys.stdout.flush()
    model = model.to(device)

    step_size = 4*batch_size*len(train_loader.dataset)
    clr = cyclical_lr(step_size, min_lr = 0.000001, max_lr = 0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
    print("Starting Learning rate is:",clr(2*step_size))
    sys.stdout.flush()
    ############## STORING THE HISTORY OF THE LOSSES #################
    avg_val_loss = 0
    total_val_loss = 0
    best_val_loss = 2000
    best_tr_loss = 2000
    total_loss = 0
    total_dice = 0
    best_idx = 0
    best_n_val_list = []
    val_avg_loss_list = []
    ################ TRAINING THE MODEL##############
    for ep in range(num_epochs):
        start = time.time()
        print("\n")
        print("Epoch Started at:", datetime.datetime.now())
        print("Epoch # : ",ep)
        print("Learning rate:", optimizer.param_groups[0]['lr'])
        model.train
        for batch_idx, (subject) in enumerate(train_loader):
            # Load the subject and its ground truth
            # read and concat the images
            image = torch.cat([batch[key][torchio.DATA] for key in batch.keys()], dim=1) # concatenate channels 
            # read the mask
            mask = batch['label'][torchio.DATA] # get the label image
            # Loading images into the GPU and ignoring the affine
            image, mask = image.float().to(device), mask.float().to(device)
            #Variable class is deprecated - parameteters to be given are the tensor, whether it requires grad and the function that created it   
            image, mask = Variable(image, requires_grad = True), Variable(mask, requires_grad = True)
            # Making sure that the optimizer has been reset
            optimizer.zero_grad()
            # Forward Propagation to get the output from the models
            torch.cuda.empty_cache()
            output = model(image.float())
            # Computing the loss
            loss = loss_fn(output.double(), mask.double(),n_classes)
            # Back Propagation for model to learn
            loss.backward()
            #Updating the weight values
            optimizer.step()
            #Pushing the dice to the cpu and only taking its value
            curr_loss = dice_loss(output[:,0,:,:,:].double(), mask[:,0,:,:,:].double()).cpu().data.item()
            #train_loss_list.append(loss.cpu().data.item())
            total_loss+=curr_loss
            # Computing the average loss
            average_loss = total_loss/(batch_idx + 1)
            #Computing the dice score 
            curr_dice = 1 - curr_loss
            #Computing the total dice
            total_dice+= curr_dice
            #Computing the average dice
            average_dice = total_dice/(batch_idx + 1)
            scheduler.step()
            torch.cuda.empty_cache()
        print("Epoch Training dice:" , average_dice)      
        if average_dice > 1-best_tr_loss:
            best_tr_idx = ep
            best_tr_loss = 1 - average_dice
        total_dice = 0
        total_loss = 0     
        print("Best Training Dice:", 1-best_tr_loss)
        print("Best Training Epoch:", best_tr_idx)
        # Now we enter the evaluation/validation part of the epoch    
        model.eval        
        for batch_idx, (subject) in enumerate(val_loader):
            with torch.no_grad():
                # image = subject['image']
                image = torch.cat([batch[key][torchio.DATA] for key in batch.keys()], dim=1) # concatenate channels 
                # mask = subject['gt']
                mask = batch['label'][torchio.DATA] # get the label image
                image, mask = image.to(device), mask.to(device)
                output = model(image.float())
                curr_loss = dice_loss(output[:,0,:,:,:].double(), mask[:,0,:,:,:].double()).cpu().data.item()
                total_loss+=curr_loss
                # Computing the average loss
                average_loss = total_loss/(batch_idx + 1)
                #Computing the dice score 
                curr_dice = 1 - curr_loss
                #Computing the total dice
                total_dice+= curr_dice
                #Computing the average dice
                average_dice = total_dice/(batch_idx + 1)

        print("Epoch Validation Dice: ", average_dice)
        torch.save(model, model_path + which_model  + str(ep) + ".pt")
        if ep > save_best:
            keep_list = np.argsort(np.array(val_avg_loss_list))
            keep_list = keep_list[0:save_best]
            for j in range(ep):
                if j not in keep_list:
                    if os.path.isfile(os.path.join(model_path + which_model  + str(j) + ".pt")):
                        os.remove(os.path.join(model_path + which_model  + str(j) + ".pt"))
            
            print("Best ",save_best," validation epochs:", keep_list)

        total_dice = 0
        total_loss = 0
        stop = time.time()   
        val_avg_loss_list.append(1-average_dice)  
        print("Time for epoch:",(stop - start)/60,"mins")    
        sys.stdout.flush()


    # ## do the actual training before this line
    # if sge_run:
    #     # call the training function as a qsub command and proceed with training in parallel
    #     test = 1 # delete this
    # else:
    #     # call the training function as a normal function and proceed with training serially
    #     test = 2 # delete this

    # check for single fold training
    if singleFoldTraining:
        break

    currentFold = currentFold + 1 # increment the fold

