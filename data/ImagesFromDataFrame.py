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
from torchio import Image

class ImagesFromDataFrame():
    """
    This class takes a pandas dataframe as input and structures it into a data structure that can be passed to Torch
    """
    def __init__(self, dataframe, psize, augmentations = None):
        self.input_df = dataframe
        self.augmentations = augmentations
        psize = self.psize

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
        im_stack,gt = self.rcrop(im_stack,gt,psize)
        gt = one_hot(gt)
        im_stack, gt = self.transform(im_stack, gt, dim)
        sample = {'image': im_stack, 'gt' : gt}
        return sample

