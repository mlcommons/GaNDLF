######START OF EMBEDDED SGE COMMANDS ##########################
#$ -S /bin/bash
#$ -cwd
#$ -N SessionName
#$ -M user.email@domain #### email to nofity with following options/scenarios
#$ -m a #### send mail in case the job is aborted
#$ -m b #### send mail when job begins
#$ -m e #### send mail when job ends
#$ -m s #### send mail when job is suspended
#$ -l h_vmem=32G
#$ -l gpu
#$ -o ./output_dir/log_\$JOB_ID.stdout 
#$ -e ./output_dir/log_\$JOB_ID.stderr
############################## END OF DEFAULT EMBEDDED SGE COMMANDS #######################
# place this in the `experiment` folder, along with the model configuration and data list
CUDA_VISIBLE_DEVICES=`get_CUDA_VISIBLE_DEVICES` || exit
export CUDA_VISIBLE_DEVICES 
source activate ../venv/
python ../deep_sage -config ./model.cfg -data ./train.csv -output ./output_dir/ -train 1 -device cuda:\$CUDA_VISIBLE_DEVICES