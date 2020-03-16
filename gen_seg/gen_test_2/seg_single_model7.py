#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:02:35 2019

@author: megh
"""
import numpy as np
import torch
import nibabel as nib
import os
import sys
import pandas as pd


data_path = "/cbica/home/bhaleram/comp_space/brets/data/test/"
save_path = "/cbica/home/bhaleram/comp_space/brets/new_scripts/ResUNet/Exp_1/gen_seg/gen_test/stored_outputs_test/"
model_path1 = "/cbica/home/bhaleram/comp_space/brets/model/ResUNet/Exp_1/mod7586.pt"

patient_name = sys.argv[1] 
dce_000 = np.expand_dims(nib.load(data_path + patient_name + "/" + patient_name + "_DCE_000_N3.nii").get_fdata(), axis = 0)
dce_001 = np.expand_dims(nib.load(data_path + patient_name + "/" + patient_name + "_DCE_001_N3.nii").get_fdata(), axis = 0)
dce_002 = np.expand_dims(nib.load(data_path + patient_name + "/" + patient_name + "_DCE_002_N3.nii").get_fdata(), axis = 0)

img = np.expand_dims(np.concatenate((dce_000,dce_001,dce_002),axis = 0), axis = 0)    
img = torch.tensor(img)

aff = nib.load(data_path + patient_name + "/" + patient_name + "_DCE_000_N3.nii").affine

model1 = torch.load(model_path1, map_location = 'cpu')

model1.cpu()
model1.eval()


seg_pred = model1(img.float())

seg = seg_pred.cpu().detach().numpy()  
seg = (seg>0.5).astype(int)
seg = seg[0,0,:,:,:]
print("Iteration done")

df = pd.read_csv("test_info.csv")
df1 = pd.read_csv("train_info.csv")
frames = [df,df1]
df = pd.concat(frames)
pid = patient_name
patient_info = df.loc[df['PatientID'] == pid]
print(patient_info)
PRmin = patient_info.iloc[0,1]
PRmax = patient_info.iloc[0,2]
PCmin = patient_info.iloc[0,3]
PCmax = patient_info.iloc[0,4]
PZmin = patient_info.iloc[0,5]
PZmax = patient_info.iloc[0,6]

x,y,z = seg.shape


seg_new = seg[PRmin:x-PRmax,PCmin:y-PCmax,PZmin:z-PZmax]

seg_new = nib.Nifti1Image(seg_new, aff)

nib.save(seg_new, save_path + patient_name + "_7.nii.gz")
