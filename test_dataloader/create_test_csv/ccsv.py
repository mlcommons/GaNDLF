import csv 
import os
import pandas as pd
#Creates a CSV file in the same folder where the experiment is being carried out

train_path = "/cbica/home/bhaleram/comp_space/brets/data/test/"
f1 = open('test.csv','w+')
patient_train_list = os.listdir(train_path)
f1.write('DCE_000_N3,DCE_001_N3,DCE_002_N3, gt\n')
for patient in patient_train_list:
    f1 = open("test.csv", 'a')
    with f1:
        writer = csv.writer(f1)
        writer.writerow([train_path + patient + '/' + patient + "_DCE_000_N3.nii", 
                         train_path + patient + '/' + patient + "_DCE_001_N3.nii",
                         train_path + patient + '/' + patient + "_DCE_002_N3.nii",
                         train_path + patient + '/' + patient + "_mask.nii"])
