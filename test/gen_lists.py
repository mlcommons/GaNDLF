import csv 
import os
import pandas as pd
#Creates a CSV file in the same folder where the experiment is being carried out
train_path = "/cbica/home/bhaleram/comp_space/brets/data/train/"
f0 = open('list_0.cfg','w+')
f1 = open('list_1.cfg','w+')
f2 = open('list_2.cfg','w+')
fmask = open('list_mask.cfg','w+')

patient_train_list = os.listdir(train_path)

for patient in patient_train_list:
    f0 = open("list_0.cfg", 'a')
    with f0:
        writer = csv.writer(f0)
        writer.writerow([train_path + patient + '/' + patient + "_DCE_000_N3.nii"])
        
for patient in patient_train_list:
    f1 = open("list_1.cfg", 'a')
    with f1:
        writer = csv.writer(f1)
        writer.writerow([train_path + patient + '/' + patient + "_DCE_001_N3.nii"])
        
        
for patient in patient_train_list:
    f2 = open("list_2.cfg", 'a')
    with f2:
        writer = csv.writer(f2)
        writer.writerow([train_path + patient + '/' + patient + "_DCE_002_N3.nii"])
        
for patient in patient_train_list:
    fmask = open("list_mask.cfg", 'a')
    with fmask:
        writer = csv.writer(fmask)
        writer.writerow([train_path + patient + '/' + patient + "_mask.nii"])

df = pd.read_csv("list_0.cfg")
df_train = df.iloc[0:117, :]
df_val = df.iloc[117:130, :]
df_train.to_csv("train_0.cfg", index = False)
df_val.to_csv("holdout_0.cfg", index = False)

df = pd.read_csv("list_1.cfg")
df_train = df.iloc[0:117, :]
df_val = df.iloc[117:130, :]
df_train.to_csv("train_1.cfg", index = False)
df_val.to_csv("holdout_1.cfg", index = False)

df = pd.read_csv("list_2.cfg")
df_train = df.iloc[0:117, :]
df_val = df.iloc[117:130, :]
df_train.to_csv("train_2.cfg", index = False)
df_val.to_csv("holdout_2.cfg", index = False)

df = pd.read_csv("list_mask.cfg")
df_train = df.iloc[0:117, :]
df_val = df.iloc[117:130, :]
df_train.to_csv("train_mask.cfg", index = False)
df_val.to_csv("holdout_mask.cfg", index = False)


