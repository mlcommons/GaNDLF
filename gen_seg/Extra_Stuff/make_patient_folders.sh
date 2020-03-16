FILES="/cbica/comp_space/bhaleram/brats/BraTS_2019_Validation/*"
for file in $FILES
do
    echo $(basename $file)
    mkdir /cbica/home/bhaleram/comp_space/fets/new_scripts/ResUNet/Semantic_Segmentation/gen_seg/stored_outputs_val/$(basename $file)
    echo done
done


