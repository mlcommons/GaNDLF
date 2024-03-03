#!/usr/bin/env bash

python -m ensurepip # ensures pip is installed in the current environment
pip install --upgrade pip
pip install wheel
pip install openvino-dev==2023.0.1 # [OPTIONAL] to generate optimized models for inference
pip install mlcube_docker          # [OPTIONAL] to deploy GaNDLF models as MLCube-compliant Docker containers
pip install medmnist==2.1.0
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
