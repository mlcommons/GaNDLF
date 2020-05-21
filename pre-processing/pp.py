import numpy as np
import os
import nibabel as nib
import math as m
def normalize(image_array):
    temp = image_array > 0
    temp_image_array = image_array[temp]
    mu = np.mean(temp_image_array)
    sig = np.std(temp_image_array)
    image_array[temp] = (image_array[temp] - mu)/sig
    return image_array

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

    padded_image_data = np.pad(img_array,((pu_x, pb_x),(pu_y,pb_y),(pu_z,pb_z)),'constant')    
      
    return padded_image_data

data_path = "/cbica/home/bhaleram/comp_space/brets/data/extra/raw/all_data/"
save_path = "/cbica/home/bhaleram/comp_space/brets/data/extra/raw/all_data_pp/"
for patient in os.listdir(data_path):

    patient_nifti = nib.load(data_path+patient+"/"+patient+"_DCE_000_N3.nii")
    aff = patient_nifti.affine
    patient_data = patient_nifti.get_fdata()
    patient_norm = normalize(patient_data) 
    patient_pad = pad(patient_norm)   
    patient_norm_nifti= nib.Nifti1Image(patient_pad,aff)
    nib.save(patient_norm_nifti,save_path + patient + "/" + patient + "_DCE_000_N3.nii")

    patient_nifti = nib.load(data_path+patient+"/"+patient+"_DCE_001_N3.nii")
    aff = patient_nifti.affine
    patient_data = patient_nifti.get_fdata()
    patient_norm = normalize(patient_data)             
    patient_pad = pad(patient_norm)
    patient_norm_nifti= nib.Nifti1Image(patient_pad,aff)
    nib.save(patient_norm_nifti,save_path + patient + "/" + patient + "_DCE_001_N3.nii")

    patient_nifti = nib.load(data_path+patient+"/"+patient+"_DCE_002_N3.nii")
    aff = patient_nifti.affine
    patient_data = patient_nifti.get_fdata()
    patient_norm = normalize(patient_data)             
    patient_pad = pad(patient_norm)
    patient_norm_nifti= nib.Nifti1Image(patient_pad,aff)
    nib.save(patient_norm_nifti,save_path + patient + "/" + patient + "_DCE_002_N3.nii")

    patient_nifti = nib.load(data_path+patient+"/"+patient+"_mask.nii")
    aff = patient_nifti.affine
    patient_data = patient_nifti.get_fdata()
    patient_pad = pad(patient_data)
    patient_norm_nifti= nib.Nifti1Image(patient_pad,aff)
    nib.save(patient_norm_nifti,save_path + patient + "/" + patient + "_mask.nii")
    
    print(patient)
