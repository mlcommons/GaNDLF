#!/bin/bash
MODEL_DIR=$1
TORCH_MODEL_DIR=$2
MODEL_NAME=$3

python convert_to_onnx.py -i $MODEL_DIR$TORCH_MODEL_DIR -o $MODEL_DIR"/onnx/" -n $MODEL_NAME

for fold in {0..4}
do
	python /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py  -m $MODEL_DIR"/onnx/"$fold"/"$MODEL_NAME"_best.onnx" --input_shape [1,3,128,128] --output_dir $MODEL_DIR"ov_models/"$fold"/"
done

