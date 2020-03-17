import csv 
import os
import pandas as pd
#Creates a CSV file in the same folder where the experiment is being carried out
train_path = "/cbica/home/bhaleram/comp_space/brets/data/test/"
f0 = open('test_0.cfg','w+')
f1 = open('test_1.cfg','w+')
f2 = open('test_2.cfg','w+')
fmask = open('test_mask.cfg','w+')

patient_train_list = os.listdir(train_path)

for patient in patient_train_list:
    f0 = open("test_0.cfg", 'a')
    with f0:
        writer = csv.writer(f0)
        writer.writerow([train_path + patient + '/' + patient + "_DCE_000_N3.nii"])
        
for patient in patient_train_list:
    f1 = open("test_1.cfg", 'a')
    with f1:
        writer = csv.writer(f1)
        writer.writerow([train_path + patient + '/' + patient + "_DCE_001_N3.nii"])
        
        
for patient in patient_train_list:
    f2 = open("test_2.cfg", 'a')
    with f2:
        writer = csv.writer(f2)
        writer.writerow([train_path + patient + '/' + patient + "_DCE_002_N3.nii"])
        
for patient in patient_train_list:
    fmask = open("test_mask.cfg", 'a')
    with fmask:
        writer = csv.writer(fmask)
        writer.writerow([train_path + patient + '/' + patient + "_mask.nii"])



