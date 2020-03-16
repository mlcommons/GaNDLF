import csv 
import os
import pandas as pd
#Creates a CSV file in the same folder where the experiment is being carried out
train_path = "/cbica/home/bhaleram/comp_space/brets/data/train/"
f1 = open('train.csv','w+')
patient_train_list = os.listdir(train_path)
f1.write('DCE_000_N3,DCE_001_N3,DCE_002_N3, gt\n')
for patient in patient_train_list:
    f1 = open("train.csv", 'a')
    with f1:
        writer = csv.writer(f1)
        writer.writerow([train_path + patient + '/' + patient + "_DCE_000_N3.nii", 
                         train_path + patient + '/' + patient + "_DCE_001_N3.nii",
                         train_path + patient + '/' + patient + "_DCE_002_N3.nii",
                         train_path + patient + '/' + patient + "_mask.nii"])
df = pd.read_csv("train.csv")
df = df.sample(frac = 1)
df1 = df.iloc[0:13, :]
df2 = df.iloc[13:26, :]
df3 = df.iloc[26:39, :]
df4 = df.iloc[39:52, :]
df5 = df.iloc[52:65, :]
df6 = df.iloc[65:78, :]
df7 = df.iloc[78:91, :]
df8 = df.iloc[91:104, :]
df9 = df.iloc[104:117, :]
df10 = df.iloc[117:130, :]

df_t1 = pd.concat((df2,df3,df4,df5,df6,df7,df8,df9,df10))
df_v1 = df1
df_t2 = pd.concat((df1,df3,df4,df5,df6,df7,df8,df9,df10))
df_v2 = df2
df_t3 = pd.concat((df1,df2,df4,df5,df6,df7,df8,df9,df10))
df_v3 = df3
df_t4 = pd.concat((df1,df2,df3,df5,df6,df7,df8,df9,df10))
df_v4 = df4
df_t5 = pd.concat((df1,df2,df3,df4,df6,df7,df8,df9,df10))
df_v5 = df5
df_t6 = pd.concat((df1,df2,df3,df4, df5,df7,df8,df9,df10))
df_v6 = df6
df_t7 = pd.concat((df1,df2,df3,df4,df5,df6,df8,df9,df10))
df_v7 = df7
df_t8 = pd.concat((df1,df2,df3,df4,df5,df6,df7,df9,df10))
df_v8 = df8
df_t9 = pd.concat((df1,df2,df3,df4,df5,df6,df7,df8,df10))
df_v9 = df9
df_t10 = pd.concat((df1,df2,df3,df4,df5,df6,df7,df8,df9))
df_v10 = df10

df_t1.to_csv("train_fold1.csv", index = False)
df_t2.to_csv("train_fold2.csv", index = False)
df_t3.to_csv("train_fold3.csv", index = False)
df_t4.to_csv("train_fold4.csv", index = False)
df_t5.to_csv("train_fold5.csv", index = False)
df_t6.to_csv("train_fold6.csv", index = False)
df_t7.to_csv("train_fold7.csv", index = False)
df_t8.to_csv("train_fold8.csv", index = False)
df_t9.to_csv("train_fold9.csv", index = False)
df_t10.to_csv("train_fold10.csv", index = False)

df_v1.to_csv("validation_fold1.csv", index = False)
df_v2.to_csv("validation_fold2.csv", index = False)
df_v3.to_csv("validation_fold3.csv", index = False)
df_v4.to_csv("validation_fold4.csv", index = False)
df_v5.to_csv("validation_fold5.csv", index = False)
df_v6.to_csv("validation_fold6.csv", index = False)
df_v7.to_csv("validation_fold7.csv", index = False)
df_v8.to_csv("validation_fold8.csv", index = False)
df_v9.to_csv("validation_fold9.csv", index = False)
df_v10.to_csv("validation_fold10.csv", index = False)



