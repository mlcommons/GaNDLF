#!/usr/bin/env bash

pip install --upgrade pip
pip install wheel
pip install openvino-dev==2022.1.0 # [OPTIONAL] to generate optimized models for inference
pip install mlcube_docker          # [OPTIONAL] to deploy GaNDLF models as MLCube-compliant Docker containers
pip install medmnist==2.1.0
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
