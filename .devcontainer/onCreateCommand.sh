#!/usr/bin/env bash

pip install --upgrade pip
pip install wheel
pip install openvino-dev==2023.0.1 # [OPTIONAL] to generate optimized models for inference
pip install mlcube_docker          # [OPTIONAL] to deploy GaNDLF models as MLCube-compliant Docker containers
pip install medmnist==2.1.0
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0 -extra-index-url https://download.pytorch.org/whl/cpu
