import numpy as np
import nibabel as nib
import os
from utils import *
import glob

path = "/cbica/home/bhaleram/comp_space/brets/new_scripts/ResUNet/Exp_1/gen_seg/gen_test/stored_outputs_test/"
save_path = '/cbica/home/bhaleram/comp_space/brets/seg_labels/ResUNet/Exp_1/'
nifti_list = os.listdir(path)
patient_list = glob.glob(path + "*_1.nii.gz")
patient_list_2 = []
for patient in patient_list:
    a  = os.path.basename(patient)
    patient_list_2.append(a)
patient_list_3 = []
for patient in patient_list_2:
    a = patient.replace("_1.nii.gz","")
    patient_list_3.append(a)
patient_list = patient_list_3
for patient in patient_list:
    print(patient)
#    seg1 = nib.load(path + patient +  "_1.nii.gz").get_fdata()
    seg2 = nib.load(path + patient + "_2.nii.gz").get_fdata()
    seg3 = nib.load(path + patient  + "_3.nii.gz").get_fdata()
    seg4 = nib.load(path + patient + "_4.nii.gz").get_fdata()
    seg5 = nib.load(path + patient  + "_5.nii.gz").get_fdata()
    seg6 = nib.load(path + patient  + "_6.nii.gz").get_fdata()
    seg7 = nib.load(path + patient  + "_7.nii.gz").get_fdata()
    seg8 = nib.load(path + patient  + "_8.nii.gz").get_fdata()
    seg9 = nib.load(path + patient  + "_9.nii.gz").get_fdata()
    aff = nib.load(path + patient   + "_1.nii.gz").affine
    seg = seg2 + seg3 + seg4 + seg5 + seg6 + seg7 + seg8 + seg9 
    seg = (seg>=4).astype(int)
    seg = nib.Nifti1Image(seg,aff)
    nib.save(seg,save_path + patient + ".nii.gz")
    print("saved")
