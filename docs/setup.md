# Setup/Installation Instructions

## Prerequisites

- Python3 with a preference for [conda](https://www.anaconda.com/)
- [CUDA](https://developer.nvidia.com/cuda-download) and a compatible [cuDNN](https://developer.nvidia.com/cudnn) installed system-wide

## Instructions

```powershell
git clone ${gandlf_repo_link}
cd GANDLF
conda create -p ./venv python=3.6.5 -y
conda activate ./venv
conda install requests pytorch torchvision cudatoolkit=10.2 -c pytorch -y # install according to your cuda version https://pytorch.org/get-started/locally/
conda install -c sdvillal openslide # this is required for windows
pip install -e .
```