#!/bin/bash

if [ $# -eq 3 ]
  then
	for data_type in "train" "validation"
	do
		for n_fold in {0..4}
		do
			python generate_data_for_quantization.py -t $data_type -n $n_fold -p $1$2 -d $1$3
		done
	done
else
	data_type="train"
	for n_fold in {0..4}
        do
              python generate_data_for_quantization.py -t $data_type -n $n_fold -p $1$2 -d $1$3 -s $4
        done
fi
