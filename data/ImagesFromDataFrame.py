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

class ImagesFromDataFrame():
  """
  Documentation for the class goes here
  """
  def __init__(self, dataFrame, augmentations):
    self.input_df = dataFrame
