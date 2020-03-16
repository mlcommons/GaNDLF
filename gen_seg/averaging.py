import numpy as np
import nibabel as nib
import os
from utils import *

path = "/cbica/home/bhaleram/comp_space/brats/new_scripts/5/gen_seg/stored_outputs_val/"
save_path = '/cbica/home/bhaleram/comp_space/brats/Segmentation_Labels_Test/seg_labels5/seg_majority_voting/'


def convert_to_3D(seg):
    seg = seg[0,:,:,:]*4 + seg[1,:,:,:]*1 + seg[2,:,:,:]*2 + seg[3,:,:,:]*0
    return seg


patient_list = os.listdir(path)
for patient in patient_list:
    seg1 = nib.load(path + patient + "1.nii.gz").get_fdata()
    seg2 = nib.load(path + patient + "2.nii.gz").get_fdata()
    seg3 = nib.load(path + patient + "3.nii.gz").get_fdata()
    seg4 = nib.load(path + patient + "4.nii.gz").get_fdata()
    seg5 = nib.load(path + patient + "5.nii.gz").get_fdata()
    aff = nib.load(path + patient + "1.nii.gz").affine
    seg = seg1 + seg2 + seg3 + seg4 + seg5
    seg = (seg>2.5).astype(int)
    seg = convert_to_3D(seg)    
    seg = nib.Nifti1Image(seg,aff)
    nib.save(seg,save_path + patient + ".nii.gz")
    print("saved")
