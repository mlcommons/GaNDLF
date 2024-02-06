#!/usr/bin/env bash

# if there is a gpu present in the container, install the gpu version of pytorch
# we will check for the presence of the nvidia-smi command as that signifies the presence of a gpu

# if runnning on a GPU machine, install the GPU version of pytorch
if command -v nvidia-smi &> /dev/null
then
	pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
fi

pip install -e .
python ./gandlf_verifyInstall
gzip -dk -r tutorials/classification_medmnist_notebook/medmnist/dataset
