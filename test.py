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
from data import TumorSegmentationDataset
from data_val import TumorSegmentationDataset_val
from schd import *
from new_models import fcn,unet,resunet
import gc
from torchsummary import summary
import nibabel as nib
from losses import *
import sys
import ast 
import datetime

parser = argparse.ArgumentParser(description = "3D Image Semantic Segmentation using Deep Learning")
parser.add_argument("-test", help = 'test configuration file')
parser.add_argument("-load", help = 'load model weight file')

parser.add_argument("-dev", help = 'choose device')
args = parser.parse_args()

test_parameters = args.test
model_path = args.load
dev = args.dev

df_test = pd.read_csv(test_parameters, sep=' = ', names=['param_name', 'param_value'],
                         comment='#', skip_blank_lines=True,
                         engine='python').fillna(' ')

#Read the parameters as a dictionary so that we can access everything by the name and so when we add some extra parameters we dont have to worry about the indexing
params = {}
for i in range(df_test.shape[0]):
    params[df_test.iloc[i, 0]] = df_test.iloc[i, 1]

# Extrating the training parameters from the dictionary
batch = int(params['batch_size'])
which_loss = params['loss_function']
channelsTe = params['channelsTesting']
channelsTr = ast.literal_eval(channelsTr) 
labelsTe = str(params['gtLabelsTesting'])
channelsVal = params['channelsValidation']
channelsVal = ast.literal_eval(channelsVal) 
psize = params['patch_size']
psize = ast.literal_eval(psize) 
psize = np.array(psize)
#Changing the channels into a proper dataframe for training data
df_final_test = pd.read_csv(channelsTe[0])
df_labels_test = pd.read_csv(labelsTe)
for channel in channelsTe:
    df = pd.read_csv(channel)
    df_final_test = pd.concat([df_final_test,df],axis=1)
df_final_test = df_final_test.drop(df.columns[[0]],axis=1)
df_final_test = pd.concat([df_final_test,df_labels_test],axis=1)

#Defining our model here according to parameters mentioned in the configuration file : 
model =  torch.load(model_path)
################################ PRINTING SOME STUFF ######################

testing_start_time = time.asctime()
startstamp = time.time()
print("\nHostname   :" + str(os.getenv("HOSTNAME")))
sys.stdout.flush()

# Setting up the train and validation loader
dataset_test = TumorSegmentationDataset_test(df_final_test,psize)
test_loader = DataLoader(dataset_test,batch_size= batch,shuffle=True,num_workers=1)


print("Testing Data Samples: ", len(test_loader.dataset))
sys.stdout.flush()
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
for batch_idx, (subject) in enumerate(test_loader):
    with torch.no_grad():
        image_1 = subject['image_1']
        image_2 = subject['image_2']
        image_3 = subject['image_3']
        image_4 = subject['image_4']

        mask = subject['gt']
        b,c,x,y,z = mask.shape
        image_1, image_2, image_3, image_4, mask = image_1.cuda(),image_2.cuda(),image_3.cuda(),image_4.cuda(), mask.cuda()

        output_1 = model(image_1.float())
        output_2 = model(image_2.float())
        output_3 = model(image_3.float())
        output_4 = model(image_4.float())

        output = torch.tensor(np.zeros((b,c,x,y,z)))

        output[:,:,0:144,0:144,:] = output_1
        output[:,:,x-144:x,0:144,:] = output_2
        output[:,:,0:144,y-144:y,:] = output_3
        output[:,:,x-144:x,y-144:y,:] = output_4
        output = output.cuda()

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
        print("Current Dice is: ", curr_dice)


print("Average dice is: ", average_dice)
