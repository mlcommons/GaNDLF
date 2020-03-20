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
from data_test import TumorSegmentationDataset_test
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
channelsTe = ast.literal_eval(channelsTe)
labelsTe = str(params['gtLabelsTesting'])
psize = params['patch_size']
psize = ast.literal_eval(psize) 
psize = np.array(psize)
save_path = str(params['path_save_seg'])
to_replace = str(params['redundant_string'])
#Changing the channels into a proper dataframe for training data
# Somehow handling the case when the GT labels are not provided

if labelsTe == ".":
    print("Working fine")
df_final_test = pd.read_csv(channelsTe[0])
if labelsTe != ".":
    df_labels_test = pd.read_csv(labelsTe)
if labelsTe == ".":
    df_labels_test = pd.read_csv(channelsTe[0])


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
total_loss = 0
average_loss = 0
total_dice = 0
average_dice = 0
for batch_idx, (subject) in enumerate(test_loader):
    with torch.no_grad():
        image = subject['image']
        mask = subject['gt']
        aff = subject['aff'].cpu().detach().numpy()
        aff = aff[0]
        pname = subject['pname']
        pname = str(os.path.basename(pname[0])).replace(to_replace,"")
        b,c,x,y,z = mask.shape
        image, mask = image.to(device), mask.to(device)
        output = model(image.float())
        if labelsTe != ".":
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
        if labelsTe == ".":
            print("Sorry, GT label not provided and  hence loss can not be calculated")

        output = output.cpu().detach().numpy()
        nib.save(nib.Nifti1Image(output,aff),save_path + pname + "_pmask.nii")
        
print("Average dice is: ", average_dice)
