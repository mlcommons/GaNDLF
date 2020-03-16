import numpy as np
import os
import nibabel as nib 
from losses import * 
import torch
# Add a forward slash after the path
seg_path = "/cbica/home/bhaleram/comp_space/brets/new_scripts/ResUNet/Exp_1/gen_seg/gen_train/stored_outputs_train/"
gt_path = "/cbica/home/bhaleram/comp_space/brets/seg_labels/test_labels/"
dice_agg = 0
count = 0
for seg_p in os.listdir(seg_path):
    seg = nib.load(seg_path + seg_p).get_fdata()
    gt = seg_p.replace("_2.nii.gz","")
    gt = nib.load(gt_path + gt + "/" + gt + "_mask.nii").get_fdata()
    seg = torch.tensor(seg)
    gt = torch.tensor(gt)
    dice_agg  = dice_agg + dice(gt,seg)
    count = count + 1
    print(dice(gt,seg))

dice = dice_agg/count
print(dice)

