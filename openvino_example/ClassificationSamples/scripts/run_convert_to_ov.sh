#!/bin/bash
MODEL_DIR=$1
TORCH_MODEL_DIR=$2
ONNX_MODEL_DIR="onnx"

python convert_to_onnx.py -i $MODEL_DIR$TORCH_MODEL_DIR -o $MODEL_DIR$ONNX_MODEL_DIR

for fold in {0..4}
do
	python /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py  -m $MODEL_DIR"onnx/"$fold"/vgg11_best.onnx" --input_shape [1,3,128,128] --output_dir $MODEL_DIR"ov_models/"$fold"/"
done

