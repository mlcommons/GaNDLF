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
class TumorSegmentationDataset_val(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
    def __len__(self):
        return len(self.df)
    def transform(self,img ,gt):
        if random.random()<0.12:
            img, gt = augment_rot90(img, gt)
            img, gt = img.copy(), gt.copy()
            print("90 degree aug done")
            print("Image shape:",img.shape)
            print("Mask shape:",gt.shape)         
        if random.random()<0.12:
            img, gt = augment_mirroring(img, gt)
            img, gt = img.copy(), gt.copy()
            print("mirroring aug done")
            print("Image shape:",img.shape)
            print("Mask shape:",gt.shape)
        if random.random()<0.12:
            img = scipy.ndimage.rotate(img,45,axes=(2,1,0),reshape=False,mode='constant')
            gt = scipy.ndimage.rotate(gt,45,axes=(2,1,0),reshape=False,order=0)        
            img, gt = img.copy(), gt.copy() 
            print("45 degree aug done") 
            print("Image shape:",img.shape)
            print("Mask shape:",gt.shape)
        if random.random()<0.12:
            img, gt = np.flipud(img).copy(),np.flipud(gt).copy()
            print("Up down flipping done")
            print("Image shape:",img.shape)
            print("Mask shape:",gt.shape)
        if random.random() < 0.12:
            img, gt = np.fliplr(img).copy(), np.fliplr(gt).copy()
            print("left right flipping done")  
            print("Image shape:",img.shape)
            print("Mask shape:",gt.shape)

        if random.random() < 0.12:
            img[0] = gaussian(img[0],True,0,0.1)   
            img[1] = gaussian(img[1],True,0,0.1)
            img[2] = gaussian(img[2],True,0,0.1)
            print("gaussian noise done")
            print("Image shape:",img.shape)
            print("Mask shape:",gt.shape)

        return img,gt
        
    def rcrop(self,imshape,psize):        
        xshift = random.randint(0,imshape[0]-psize[0])
        yshift = random.randint(0,imshape[1]-psize[1]) 
        #zshift = random.randint(0,imshape[2]-psize[2])
        return xshift, yshift

    def __getitem__(self, index):
        
        dce_000_path = self.df.iloc[index, 0]
        dce_001_path = self.df.iloc[index, 1]
        dce_002_path = self.df.iloc[index, 2]
        gt_path = self.df.iloc[index,3]
        print(dce_000_path)
        dce_000 = nib.load(dce_000_path).get_fdata()
        dce_001 = nib.load(dce_001_path).get_fdata()
        dce_002 = nib.load(dce_002_path).get_fdata()
        gt = nib.load(gt_path).get_fdata()
            
        psize = (64,64,64)
        
        xshift, yshift= self.rcrop(dce_000.shape,psize)
        print("before patch",dce_000.shape)
        #print(xshift,yshift)
        dce_000 = dce_000[xshift:xshift+psize[0],yshift:yshift+psize[1],:]
        dce_001 = dce_001[xshift:xshift+psize[0],yshift:yshift+psize[1],:]
        dce_002 = dce_002[xshift:xshift+psize[0],yshift:yshift+psize[1],:]
        gt = gt[xshift:xshift+psize[0],yshift:yshift+psize[1],:]
        dce_000 = np.expand_dims(dce_000,axis = 0)
        dce_001 = np.expand_dims(dce_001,axis = 0)
        dce_002 = np.expand_dims(dce_002,axis = 0)
        
        image = np.concatenate((dce_000,dce_001,dce_002),axis = 0)      
        gt = np.expand_dims(gt, axis = 0)

        print("before transform",image.shape)
        image, gt = self.transform(image, gt)
        print("after transform",image.shape)
        #print("Image shape",image.shape)
        #print("Gt Shape",gt.shape)

            
        print(xshift,yshift)
        #print(dce_000.shape)
        sample = {'image': image, 'gt' : gt}
        print("Inside dataloader",sample['image'].shape)
        return sample



