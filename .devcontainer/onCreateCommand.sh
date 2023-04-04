#!/usr/bin/env bash


pip install --upgrade pip
pip install wheel
pip install openvino-dev==2022.1.0 # [OPTIONAL] to generate optimized models for inference
pip install mlcube_docker # [OPTIONAL] to deploy GaNDLF models as MLCube-compliant Docker containers
