###############START OF EMBEDDED SGE COMMANDS ##########################
#$ -S /bin/bash
#$ -cwd
#$ -N cpujob
#$ -M megh.bhalerao@gmail.com #### email to nofity with following options/scenarios
#$ -m a #### send mail in case the job is aborted
#$ -m b #### send mail when job begins
#$ -m e #### send mail when job ends
#$ -m s #### send mail when job is suspended
#$ -l h_vmem=100G
#$ -l cpu
############################## END OF DEFAULT EMBEDDED SGE COMMANDS #######################
module load python/anaconda/3
module load pytorch/1.0.1
module unload gcc
module load gcc/5.2.0

FILES='/cbica/home/bhaleram/comp_space/brats/BraTS_2019_Validation/*'
for file in $FILES
do
    echo $(basename $file)
    python seg_patch.py $(basename $file)
    echo done
done

