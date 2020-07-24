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
from torchio.transforms import *
from torchio import Image, Subject
from sklearn.model_selection import KFold
from data.ImagesFromDataFrame import ImagesFromDataFrame

# Defining a dictionary - key is the string and the value is the augmentation object
## todo: ability to change interpolation type from config file
global_augs_dict = {
    'affine':RandomAffine(image_interpolation = 'linear'), 
    'elastic': RandomElasticDeformation(num_control_points=(7, 7, 7),locked_borders=2),
    'motion': RandomMotion(degrees=10, translation = 10, num_transforms= 2, image_interpolation = 'linear', p = 1., seed = None), 
    'ghosting': RandomGhosting(num_ghosts = (4, 10), axes = (0, 1, 2), intensity = (0.5, 1), restore = 0.02, p = 1., seed = None),
    'bias': RandomBiasField(coefficients = 0.5, order= 3, p= 1., seed = None), 
    'blur': RandomBlur(std = (0., 4.), p = 1, seed = None), 
    'noise':RandomNoise(mean = 0, std = (0, 0.25), p = 1., seed = None) , 
    'swap':RandomSwap(patch_size = 15, num_iterations = 100, p = 1, seed = None) 
}


# This function takes in a dataframe, with some other parameters and returns the dataloader
def Trainer(dataframe, parameters):

  kfolds = int(parameters['kcross_validation'])
  test = 1