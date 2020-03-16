FILES="/cbica/comp_space/bhaleram/brats/data/train_all/*"
for file in $FILES
do
    echo $(basename $file)
    mkdir /cbica/home/bhaleram/comp_space/brats/new_scripts/31/gen_seg/stored_outputs_train/$(basename $file)
    echo done
done


