# How are the segmentations generated on unknown data?
Again, ideally, the documentation of this entire repo is written under the assumption that the user is working on a SGE based HPC cluster. 

Also, additionally , it is always a good practice to **avoid storing stuff other than codes/scripts in github repo**. Doing this avoids problems in version control. 

1. This is the folder you want to `cd` into when you want to generate tumor segmentations for the testing datset (preprocessed).
2. The segmentation is generated using 5 models (which are trained using 5 fold cross validation), by first individually generating predictions of each model and then combining them using majority voting.
3. In this folder you can see the files named as `seg_single_model*.*` and `submit_single*.sh`. The latter files are helper scripts for the inference and takes a patient name as input (it is the only input that this executable takes). This patient name needs to be present in `data_path` (how to set the `data_path` will be described in further points)
4. The `seg_single_model*.py` needs to be edited with proper paths to generate the segmentations correctly.
5. Open the `seg_single_model*.py` files.
6. Set the `data_path` variable to the path where the testing preprocessed data is present (don't forget the forward slash after the path)
7. Set the `save_path` to the path where you want to save the segmentations of the individual folds. Let us call this folder `stored_outputs_test`.

**TODO : MAKE LOG FILES LEGIBLE**

**TODO:VERBOSE OPTION IN LOG FILES AND TRAIN PARAMATERS AS A COMMAND LINE ARG**

8. Now we have to set the last parameter which is the `model_path*`.
9. The `model_pathx` where x is from 1-5, is the model path to the best model of each fold (since we save `save_best` - which is usually 5 in all our exps)
10. As mentioned earlier, the models are saved as `.pt` files, so the `model_path*` will end with `modxxx.pt`.
11. Now, let's get to how to exactly choose the path to the model (weight) file. 
12. So, once you run the model for whatever number of epochs (by submitting the training scripts to the cluster as mentioned here : https://github.com/meghbhalerao/Semantic_Segmentation/tree/master/submission_scripts), the stdout and stderr files are generated with the names like $jobname.o$jobid and $jobname.e$jobid respectively.
13. You need to look manually into each of the stdout files (corresponding to each fold) to find at which epoch did the best validation-during-training loss occur and the model saved at this epoch is considered to be the 'best model' that we have been talking about before.
14. So, once you open (or cat) either of the stdout files, at the bottom of the file you can see clearly mentioned the best 5 valdation epochs.
15. Look at the number of the best epoch and that goes in the `xxx` field in the `modxxx.pt`.This is how you set the `model_pathx`. 
16. Do the above steps in the files corresponding to each fold.
17. Run each of the `seg_single_model*.sh` either by `bash` or `qsub`. These are the main points of entry for the inference. 
18. Running these bash scripts will submit `submit_single*.sh` script to the cluster multiple times (each patient in the testing data folder) with the patient name as the paramter to the script (these patient names are taken from the folder names that are traversed through in the previous script).
19. The entire patient image is segmented at once and not patch-wise (like the training process) and hence the memory requirement is high and hence the inference can't be done on a GPU.
20. Further details of the memory requirements can be found in either of the `submit_single*.sh` scripts.
21. For further details on how and where the predicted segmentations are stored (of each fold) please `cd` into the `stored_outputs_*` folder under `gen_seg`
22. Once you have understood how the 5 segmentations from the 5 folds are generated, now we can fuse all the 5 segmentations using majority voting. 
23. Open the `majority_voting.py` and change the `path` variable to the path to the folder `stored_outputs_test` and the `save_path` variable to the folder where you want to store the final segmentations.
24. Once you have made these changes run the `majority_voting.sh` script which runs the `majority_voting.py` file. 
25. Running this script will combine the predictions of the 5 folds using majority voting and save them according to the paths mentioned 
