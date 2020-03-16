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
    def __init__(self,df,psize):
        self.df = df
        self.psize = psize
    def __len__(self):
        return len(self.df)
    def transform(self,img ,gt, dim):
        if random.random()<0.12:
            img, gt = augment_rot90(img, gt)
            img, gt = img.copy(), gt.copy()         
        if random.random()<0.12:
            img, gt = augment_mirroring(img, gt)
            img, gt = img.copy(), gt.copy()
        if random.random()<0.12:
            img = scipy.ndimage.rotate(img,45,axes=(2,1,0),reshape=False,mode='constant')
            gt = scipy.ndimage.rotate(gt,45,axes=(2,1,0),reshape=False,order=0) 
            img, gt = img.copy(), gt.copy() 
        if random.random()<0.12:
            img, gt = np.flipud(img).copy(),np.flipud(gt).copy()
        if random.random() < 0.12:
            img, gt = np.fliplr(img).copy(), np.fliplr(gt).copy()
        if random.random() < 0.12:
            for n in range(dim-1):
                img[n] = gaussian(img[n],True,0,0.1)   

        return img,gt
        
    def rcrop(self,image,gt,psize):
        imshape = image.shape
        if imshape[1]>psize[0]:
            xshift = random.randint(0,imshape[0]-psize[0])
            image = image[:,xshift:xshift+psize[0],:,:]
            gt = gt[xshift:xshift+psize[0],:,:]
        if imshape[2]>psize[1]:
            yshift = random.randint(0,imshape[1]-psize[1])
            image = image[:,:,yshift:yshift+psize[1],:]
            gt = gt[:,yshift:yshift+psize[1],:]
        if imshape[3]>psize[2]:
            zshift = random.randint(0,imshape[2]-psize[2])
            image = image[:,:,:,zshift:zshift+psize[2]]
            gt = gt[:,:,zshift:zshift+psize[2]]
        return image,gt

    def __getitem__(self, index):
        psize = self.psize
        imshape = nib.load(self.df.iloc[index,0]).get_fdata().shape
        dim = self.df.shape[1]
        im_stack =  np.zeros((dim-1,*imshape),dtype=int)
        for n in range(0,dim-1):
            image = self.df.iloc[index,n]
            image = nib.load(image).get_fdata()
            im_stack[n] = image                
        gt = nib.load(self.df.iloc[index,dim-1]).get_fdata()
        print(im_stack.shape)
        im_stack,gt = self.rcrop(im_stack,gt,psize)
        im_stack, gt = self.transform(im_stack, gt, dim)
        gt = one_hot(gt)
        sample = {'image': im_stack, 'gt' : gt}
        return sample



