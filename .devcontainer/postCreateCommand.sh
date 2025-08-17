#!/usr/bin/env bash

# if there is a gpu present in the container, install the gpu version of pytorch
# we will check for the presence of the nvidia-smi command as that signifies the presence of a gpu

# pip install uv  ## potentially creating permission issues
# if runnning on a GPU machine, install the GPU version of pytorch
if command -v nvidia-smi &> /dev/null
then
	pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
	# uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu --system
fi

### temp fix for monai to prevent it from pullin all nvidia stuff
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/Project-MONAI/MONAI.git
cd MONAI
pip install -e .
cd ..
### temp fix for monai to prevent it from pullin all nvidia stuff
pip install -e .
# uv pip install -e . --system
gandlf verify-install
gzip -dk -r tutorials/classification_medmnist_notebook/medmnist/dataset
