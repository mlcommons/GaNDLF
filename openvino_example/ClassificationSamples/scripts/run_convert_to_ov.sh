#!/bin/bash
ROOT_DIR=$1
TORCH_MODEL_DIR="DFU_experiments_vgg11_5fold_without_preprocess/"
ONNX_MODEL_DIR="onnx"

python convert_to_onnx.py -i $ROOT_DIR$TORCH_MODEL_DIR -o $ROOT_DIR$ONNX_MODEL_DIR

for fold in {0..4}
do
	python /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py  -m $ROOT_DIR"onnx/"$fold"/vgg11_best.onnx" --input_shape [1,3,128,128] --output_dir $ROOT_DIR"ov_models/"$fold"/"
done

