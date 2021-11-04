#!/bin/bash

if [ $# -eq 0 ]
  then
	for data_type in "train" "validation"
	do
		for n_fold in 0 1 2 3 4 
		do
			python generate_data_for_quantization.py -t $data_type -n $n_fold
		done
	done
else
	data_type="train"
	for n_fold in 0 1 2 3 4
        do
              python generate_data_for_quantization.py -t $data_type -n $n_fold -s $1
        done
fi
