from __future__ import print_function, division
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
from data import *
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

parser = argparse.ArgumentParser(description = "3D Image Semantic Segmentation using Deep Learning")
parser.add_argument("-model", type=str, help = 'model configuration file', required=True)
parser.add_argument("-data", type=str, help = 'data csv file that is used for training or testing', required=True)
parser.add_argument("-output", type=str, help = 'output directory to save intermediate files and model weights', required=True)
parser.add_argument("-train", default=1, type=int, help = '1 means training and 0 means testing; for 0, there needs to be a compatible checkpoint saved in \'output\'', required=False)
parser.add_argument("-dev", default=0, type=int, help = 'choose device', required=True)
args = parser.parse_args()

file_trainingData_full = args.data
model_parameters = args.model
dev = args.dev
model_path = args.output
mode = args.train

# safe directory creation
Path(model_path).mkdir(parents=True, exist_ok=True)

# df_train = pd.read_csv(train_parameters, sep=' = ', names=['param_name', 'param_value'],
#                          comment='#', skip_blank_lines=True,
#                          engine='python').fillna(' ')

df_model = pd.read_csv(model_parameters, sep=' = ', names=['param_name', 'param_value'],
                         comment='#', skip_blank_lines=True,
                         engine='python').fillna(' ')

#Read the parameters as a dictionary so that we can access everything by the name and so when we add some extra parameters we dont have to worry about the indexing
params = {}
# for i in range(df_train.shape[0]):
#     params[df_train.iloc[i, 0]] = df_train.iloc[i, 1]

for j in range(df_model.shape[0]):
    params[df_model.iloc[j, 0]] = df_model.iloc[j, 1]

# Extrating the training parameters from the dictionary
num_epochs = int(params['num_epochs'])
batch = int(params['batch_size'])
learning_rate = int(params['learning_rate'])
which_loss = params['loss_function']
opt = str(params['opt'])
save_best = int(params['save_best'])
# Defining the channels for training and validation
# channelsTr = params['channelsTraining']
# channelsTr = ast.literal_eval(channelsTr) 
# labelsTr = str(params['gtLabelsTraining'])
# channelsVal = params['channelsValidation']
# channelsVal = ast.literal_eval(channelsVal) 
# labelsVal = str(params['gtLabelsValidation'])

# Extracting the model parameters from the dictionary
n_classes = int(params['numberOfOutputClasses'])
base_filters = int(params['base_filters'])
n_channels = int(params['numberOfInputChannels'])
model_path = str(params['folderForOutput'])
which_model = str(params['modelName'])
psize = params['patch_size']
psize = ast.literal_eval(psize) 
psize = np.array(psize)

## read training dataset into data frame
trainingData_full = pd.read_csv(file_trainingData_full)

## contruct the training and validation data from trainingCSV

#Changing the channels into a proper dataframe for training da
df_final_train = pd.DataFrame() # initialize the variable to hold the training channels
df_labels_train = pd.read_csv(labelsTr)
for channel in channelsTr:
    df = pd.read_csv(channel)
    df_final_train = pd.concat([df_final_train,df],axis=1)
df_final_train = pd.concat([df_final_train,df_labels_train],axis=1)

#Changing the channels into a proper dataframe for Validation data
df_final_val = pd.DataFrame()
df_labels_val = pd.read_csv(labelsVal)
for channel in channelsVal:
    df = pd.read_csv(channel)
    df_final_val = pd.concat([df_final_val,df],axis=1)
df_final_val = pd.concat([df_final_val,df_labels_val],axis=1)

#Defining our model here according to parameters mentioned in the configuration file : 
if which_model == 'resunet':
    model = resunet(n_channels,n_classes,base_filters)
if which_model == 'unet':
    model = unet(n_channels,n_classes,base_filters)
if which_model == 'fcn':
    model = fcn(n_channels,n_classes,base_filters)
if which_model == 'uinc':
    model = uinc(n_channels,n_classes,base_filters)
else:
    print('WARNING: Could not find the requested model \'' + which_model + '\' in the impementation, using ResUNet, instead', file = sys.stderr)
    which_model = 'resunet'
    model = resunet(n_channels,n_classes,base_filters)

################################ PRINTING SOME STUFF ######################

training_start_time = time.asctime()
startstamp = time.time()
print("\nHostname   :" + str(os.getenv("HOSTNAME")))
sys.stdout.flush()

# Setting up the train and validation loader
dataset_train = TumorSegmentationDataset(df_final_train,psize)
train_loader = DataLoader(dataset_train,batch_size= batch,shuffle=True,num_workers=1)
dataset_valid = TumorSegmentationDataset_val(df_final_val,psize)
val_loader = DataLoader(dataset_valid, batch_size=1,shuffle=True,num_workers = 1)

print("Training Data Samples: ", len(train_loader.dataset))
sys.stdout.flush()
print(dev)
device = torch.device(dev)
print("Current Device : ", torch.cuda.current_device())
print("Device Count on Machine : ", torch.cuda.device_count())
print("Device Name : ", torch.cuda.get_device_name(device))
print("Cuda Availibility : ", torch.cuda.is_available())
print('Using device:', device)
if device.type == 'cuda':
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1),
          'GB')
    print('Cached: ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')

sys.stdout.flush()
model = model.to(device)
##################### SETTING THE OPTIMIZER ########################
if opt == 'sgd':
    optimizer = optim.SGD(model.parameters(),
                               lr= learning_rate,
                               momentum = 0.9)
if opt == 'adam':    
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9,0.999), weight_decay = 0.00005)

step_size = 4*batch*len(train_loader.dataset)
clr = cyclical_lr(step_size, min_lr = 0.000001, max_lr = 0.001)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
print("Starting Learning rate is:",clr(2*step_size))
sys.stdout.flush()
############### CHOOSING THE LOSS FUNCTION ###################
if which_loss == 'dc':
    loss_fn  = MCD_loss
if which_loss == 'dcce':
    loss_fn  = DCCE
if which_loss == 'ce':
    loss_fn = CE
if which_loss == 'mse':
    loss_fn = MCD_MSE_loss
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
        image = subject['image']
        mask = subject['gt']
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
            image = subject['image']
            mask = subject['gt']
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
