# Setup/Installation Instructions

## Prerequisites

- Python3 with a preference for [conda](https://www.anaconda.com/)
- [CUDA](https://developer.nvidia.com/cuda-download) and a compatible [cuDNN](https://developer.nvidia.com/cudnn) installed system-wide

## Installation

```powershell
conda create -n venv_gandlf python=3.6 -y
conda activate venv_gandlf
conda install -c conda-forge mamba # allows for faster dependency solving
mamba install -c pytorch pytorch torchvision -y # 1.8.0 installs cuda 10.2 by default, personalize based on your cuda/driver availability via https://pytorch.org/get-started/locally/
mamba install -c conda-forge gandlf -y
```

**Note for Windows users**: Please follow instructions for [developers](./extending).
