#!/usr/bin/env bash

# if there is a gpu present in the container, install the gpu version of pytorch
# we will check for the presence of the nvidia-smi command as that signifies the presence of a gpu

# if runnning on a GPU machine, install the GPU version of pytorch
if command -v nvidia-smi &> /dev/null
then
	pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
fi

pip install -e .
gandlf verify-install
gzip -dk -r tutorials/classification_medmnist_notebook/medmnist/dataset
