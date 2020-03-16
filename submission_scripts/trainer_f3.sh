######START OF EMBEDDED SGE COMMANDS ##########################
#$ -S /bin/bash
#$ -cwd
#$ -N brets_res_f3
#$ -M megh.bhalerao@gmail.com #### email to nofity with following options/scenarios
#$ -m a #### send mail in case the job is aborted
#$ -m b #### send mail when job begins
#$ -m e #### send mail when job ends
#$ -m s #### send mail when job is suspended
#$ -l h_vmem=32G
#$ -l gpu
############################## END OF DEFAULT EMBEDDED SGE COMMANDS #######################
CUDA_VISIBLE_DEVICES=`get_CUDA_VISIBLE_DEVICES` || exit
export CUDA_VISIBLE_DEVICES 

module load pytorch/1.0.1
module load python/anaconda/3
module unload gcc
module load gcc/5.2.0
cd ..
python trainer.py /cbica/home/bhaleram/comp_space/brets/new_scripts/ResUNet/Exp_1/csv_all/train_fold3.csv /cbica/home/bhaleram/comp_space/brets/new_scripts/ResUNet/Exp_1/csv_all/validation_fold3.csv
