import numpy as np
import os
import nibabel as nib 
from losses import * 
# Add a forward slash after the path
seg_path = "/cbica/home/bhaleram/comp_space/brets/seg_labels/ResUNet/Exp_1/"
gt_path = "/cbica/home/bhaleram/comp_space/brets/seg_labels/test_labels/"
dice_agg = 0
count = 0
for seg_p in os.listdir(seg_path):
    seg = nib.load(seg_path + seg_p)
    print(seg.shape)
    gt = seg_p.replace("_2.nii.gz","")
    gt = nib.load(gt_path + gt + "/" + gt + "_mask.nii")
    print(gt.shape)

