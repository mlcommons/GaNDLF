FILES="/cbica/home/bhaleram/comp_space/fets/data/PreProcessed_Data/brats_test/*"
for file in $FILES
do
    echo $(basename $file)
    mkdir /cbica/home/bhaleram/comp_space/fets/new_scripts/ResUNet/Semantic_Segmentation/gen_seg/stored_outputs_test/$(basename $file)
    vi /cbica/home/bhaleram/comp_space/fets/new_scripts/ResUNet/Semantic_Segmentation/gen_seg/stored_outputs_test/$(basename $file)/a.txt
    echo done
done


