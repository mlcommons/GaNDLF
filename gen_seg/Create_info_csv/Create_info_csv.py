#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:07:06 2019

@author: megh
"""

import os
import glob
import numpy as np
import math as m
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
import sys
import csv

def pad(img_array):
    pu_x = pb_x = pu_y = pb_y = pu_z = pb_z = 0
    dim = img_array.shape
    dimen = dim[0]
    if dimen%16 != 0:
        x = 16*(m.trunc(dimen/16)+1)
        pu_x = m.trunc((x - dimen)/2)
        pb_x = x - dimen - pu_x
        
    dimen = dim[1]
    
    if dimen%16 != 0:
        x = 16*(m.trunc(dimen/16)+1)
        pu_y = m.trunc((x - dimen)/2)
        pb_y = x - dimen - pu_y
        
    dimen = dim[2]
    
    if dimen%16 != 0:
        x = 16*(m.trunc(dimen/16)+1)
        pu_z = m.trunc((x - dimen)/2)
        pb_z = x - dimen - pu_z
    
    return pu_x, pb_x, pu_y, pb_y, pu_z, pb_z
    
f1 = open('test_info.csv','w+')
path_data = "/cbica/home/bhaleram/comp_space/brets/data/extra/raw/test/"
patient_list = os.listdir(path_data)
f1.write('PatientID,PRmin,PRmax,PCmin,PCmax,PZmin,PZmax \n')
f1 = open('test_info.csv','a')
for patient in patient_list:
    patient_path =  path_data + patient
    dce_000_data = nib.load(patient_path + "/" + patient + "_DCE_000_N3.nii").get_fdata()
    pu_x, pb_x, pu_y, pb_y, pu_z, pb_z = pad(dce_000_data)
    print(pu_x, pb_x, pu_y, pb_y, pu_z, pb_z) 
    f1 = open('test_info.csv','a')
    with f1:
        w = csv.writer(f1)
        w.writerow([patient,pb_x,pu_x,pb_y,pu_y,pb_z,pu_z])
        



