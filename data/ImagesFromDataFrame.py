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
import torchio
from torchio.transforms import RandomAffine, RandomElasticDeformation, Compose
from torchio import Image, Subject

# This function takes in a dataframe, with some other parameters and returns the dataloader
def ImagesFromDataFrame(dataframe, psize, augmentations = None):
    # Finding the dimension of the dataframe for computational purposes later
    num_row, num_col = dataframe.shape
    num_channels = num_col - 1
    # changing the column indices to make it easier
    dataframe.columns = range(0,num_col)
    dataframe.index = range(0,num_row)
    # This list will later contain the list of subjects 
    subjects_list = []

    # iterating through the dataframe
    for patient in range(num_row):
        # We need this dict for storing the meta data for eac subject such as different image modalities, labels, any other data
        subject_dict = {}
        # iterating through the channels/modalities/timepoint of the image
        for channel in range(num_channels):
            dataframe[channel][patient]
            

    
    imshape = nib.load(self.df.iloc[index,0]).get_fdata().shape
    dim = self.df.shape[1]
    im_stack =  np.zeros((dim-1,*imshape),dtype=int)
    for n in range(0,dim-1):
        image = self.df.iloc[index,n]
        image = nib.load(image).get_fdata()
        im_stack[n] = image
    gt = nib.load(self.df.iloc[index,dim-1]).get_fdata()
    im_stack,gt = self.rcrop(im_stack,gt,psize)
    gt = one_hot(gt)
    im_stack, gt = self.transform(im_stack, gt, dim)
    sample = {'image': im_stack, 'gt' : gt}
    return sample

