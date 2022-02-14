#!/bin/bash

export LD_LIBRARY_PATH=/home/bduser/miniconda3/envs/gandlf_tests_v14/lib:$LD_LIBRARY_PATH

NUM_CORES=56
NUM_CORES_PER_SOCK=28
NUM_SOCKETS=2

total_num_workers=2 
 
echo "Total Cores: $NUM_CORES"
echo "Num Workers: $total_num_workers"
 
ht=$(lscpu | grep "Thread(s) per core:" | awk '{print $NF}')

num_cores_per_worker=$(($NUM_CORES/$total_num_workers))
echo "Num cores per worker:"$num_cores_per_worker

for t in 8
  do
    for ((i=0;i<$total_num_workers;i++));
      do 
         phy_core_start=$(($i*$num_cores_per_worker)) 
         log_core_start=$((($i*$num_cores_per_worker)+$NUM_CORES)) 
    
         taskset_phy_core_start=$phy_core_start
         taskset_phy_core_end=("$(($phy_core_start+$t-1))")
         taskset_log_core_start=$log_core_start
         taskset_log_core_end=("$(($log_core_start+$t-1))")
         echo "Starting script with physical core ids:"$taskset_phy_core_start"-"$taskset_phy_core_end" for worker:"$i" Stepping id:"$t
         echo "Starting script with logical core ids:"$taskset_log_core_start"-"$taskset_log_core_end" for worker:"$i" Stepping id:"$t
         taskset -c $taskset_phy_core_start"-"$taskset_phy_core_end,$taskset_log_core_start"-"$taskset_log_core_end python benchmark_pt_ov.py -m 'resunet' -md './infer_models'  -ptm 3dresunet_pt -ovm 3dresunet_ov -p ./3dunet_exp/test_data_dir/parameters.pkl  -d ./3dunet_exp/tcga-val-data-pre-ma-test.csv  -o ./3dunet_exp/test_data_dir -v False 2>&1 | tee logs/3dresunet_infer_all_icx_numcores_${t}_wk_${i}.log &
         # echo "taskset -c $taskset_phy_core_start"-"$taskset_phy_core_end,$taskset_log_core_start"-"$taskset_log_core_end python benchmark_pt_ov.py -m 'resunet' -md './infer_models'  -ptm 3dresunet_pt -ovm 3dresunet_ov -p ./3dunet_exp/test_data_dir/parameters.pkl  -d ./3dunet_exp/tcga-val-data-pre-ma-test.csv  -o ./3dunet_exp/test_data_dir -v False 2>&1 | tee logs/3dresunet_infer_all_icx_numcores_${t}_wk_${i}.log &"
    done;
    sleep 1200
done;
