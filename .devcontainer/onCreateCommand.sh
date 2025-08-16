#!/usr/bin/env bash

python -m ensurepip # ensures pip is installed in the current environment
pip install --upgrade pip==24.0
# pip install uv ## this is giving `Permission denied (os error 13)``
# uv pip install openvino-dev==2023.0.1 --system # [OPTIONAL] to generate optimized models for inference
# uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu --system
# uv pip install medmnist==2.1.0 --system
pip install openvino-dev==2023.0.1 # [OPTIONAL] to generate optimized models for inference
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install medmnist==2.1.0
