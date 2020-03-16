import csv 
import os
import pandas as pd

#Creates a CSV file in the same folder where the experiment is being carried out
test_path = "/cbica/home/bhaleram/comp_space/brets/data/test/"
f1 = open('test.csv','w+')
patient_test_list = os.listdir(test_path)
f1.write('DCE_000_N3,DCE_001_N3,DCE_002_N3, gt\n')
for patient in patient_test_list:
    f1 = open("test.csv", 'a')
    with f1:
        writer = csv.writer(f1)
        writer.writerow([test_path + patient + '/' + patient + "_DCE_000_N3.nii", 
                         test_path + patient + '/' + patient + "_DCE_001_N3.nii",
                         test_path + patient + '/' + patient + "_DCE_002_N3.nii",
                         test_path + patient + '/' + patient + "_mask.nii"])
df = pd.read_csv("test.csv")
df.to_csv("test.csv", index = False)
