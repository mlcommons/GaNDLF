from augmentations.augs import *
from augmentations.color_aug import *
from augmentations.noise_aug import *
from augmentations.spatial_augs import *
from augmentations.utils import *
import nibabel as nib
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
import random
from all_augmentations import *
from utils import *
import random
import scipy

class TumorSegmentationDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        dce_000_path = self.df.iloc[index, 0]   
        dce_001_path = self.df.iloc[index, 1]
        dce_002_path = self.df.iloc[index, 2]
        gt_path = self.df.iloc[index,3]

        dce_000 = nib.load(dce_000_path).get_fdata()        
        dce_001 = nib.load(dce_001_path).get_fdata()
        dce_002 = nib.load(dce_002_path).get_fdata()
        gt = nib.load(gt_path).get_fdata()
            
        psize = (144,144,64)
        x,y,z = gt.shape

        # Cropping the image according to the patch size

        dce_000_1 = dce_000[0:0+psize[0],0:0+psize[1],:]        
        dce_001_1 = dce_001[0:0+psize[0],0:0+psize[1],:]
        dce_002_1 = dce_002[0:0+psize[0],0:0+psize[1],:]

        dce_000_2 = dce_000[x-psize[0]:x,0:0+psize[1],:]        
        dce_001_2 = dce_001[x-psize[0]:x,0:0+psize[1],:]
        dce_002_2 = dce_002[x-psize[0]:x,0:0+psize[1],:]

        dce_000_3 = dce_000[0:0+psize[0], y-psize[1]:y,:]        
        dce_001_3 = dce_001[0:0+psize[0], y-psize[1]:y,:]
        dce_002_3 = dce_002[0:0+psize[0], y-psize[1]:y,:]


        dce_000_4 = dce_000[x-psize[0]:x,y-psize[1]:y,:]        
        dce_001_4 = dce_001[x-psize[0]:x,y-psize[1]:y,:]
        dce_002_4 = dce_002[x-psize[0]:x,y-psize[1]:y,:]
        
        # Expanding the dimsions of the images for upcoming concatenation 

        dce_000_1 = np.expand_dims(dce_000_1,axis = 0)
        dce_001_1 = np.expand_dims(dce_001_1,axis = 0)
        dce_002_1 = np.expand_dims(dce_002_1,axis = 0)
        
        dce_000_2 = np.expand_dims(dce_000_2,axis = 0)
        dce_001_2 = np.expand_dims(dce_001_2,axis = 0)
        dce_002_2 = np.expand_dims(dce_002_2,axis = 0)

        dce_000_3 = np.expand_dims(dce_000_3,axis = 0)
        dce_001_3 = np.expand_dims(dce_001_3,axis = 0)
        dce_002_3 = np.expand_dims(dce_002_3,axis = 0)

        dce_000_4 = np.expand_dims(dce_000_4,axis = 0)
        dce_001_4 = np.expand_dims(dce_001_4,axis = 0)
        dce_002_4 = np.expand_dims(dce_002_4,axis = 0)

        # Concatenating the images
        image_1 = np.concatenate((dce_000_1,dce_001_1,dce_002_1),axis = 0)      
        image_2 = np.concatenate((dce_000_2,dce_001_2,dce_002_2),axis = 0)   
        image_3 = np.concatenate((dce_000_3,dce_001_3,dce_002_3),axis = 0)   
        image_4 = np.concatenate((dce_000_4,dce_001_4,dce_002_4),axis = 0)         
        
        
        gt = one_hot(gt)

        sample = {'image_1': image_1, 'image_2': image_2, 'image_3': image_3, 'image_4': image_4, 'gt' : gt}
        return sample



