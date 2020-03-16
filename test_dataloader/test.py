from __future__ import print_function, division
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from new_models import *
import gc
import nibabel as nib
from losses import *
import sys
import datetime

test_csv = sys.argv[1]

model = torch.load("/cbica/home/bhaleram/comp_space/brets/model/ResUNet/Exp_1/mod2404.pt")

############################## PRINTING SOME STUFF ######################
training_start_time = time.asctime()
startstamp = time.time()
print("\nHostname   :" + str(os.getenv("HOSTNAME")))
sys.stdout.flush()
# Setting up the train and validation loader
dataset_test = TumorSegmentationDataset(test_csv)
test_loader = DataLoader(dataset_test,batch_size=1,shuffle=False,num_workers=4)

print("Test Data Samples: ", len(test_loader.dataset))
sys.stdout.flush()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device : ", torch.cuda.current_device())
print("Device Count on Machine : ", torch.cuda.device_count())
print("Device Name : ", torch.cuda.get_device_name(device))
print("Cuda Availibility : ", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1),
          'GB')
    print('Cached: ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')

sys.stdout.flush()
model.cuda()
############## STORING THE HISTORY OF THE LOSSES #################
total_loss = 0
total_dice = 0 
################ TESTING THE MODEL##############
model.eval
model.cuda()      
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
