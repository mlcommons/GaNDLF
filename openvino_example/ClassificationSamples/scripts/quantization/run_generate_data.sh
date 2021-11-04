#!/bin/bash

if [ $# -eq 1 ]
  then
	for data_type in "train" "validation"
	do
		for n_fold in {0..4}
		do
			python generate_data_for_quantization.py -t $data_type -n $n_fold -r $1
		done
	done
else
	data_type="train"
	for n_fold in {0..4}
        do
              python generate_data_for_quantization.py -t $data_type -n $n_fold -r $1 -s $2
        done
fi
