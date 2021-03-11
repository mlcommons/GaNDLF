# Setup/Installation Instructions

## Prerequisites

- Python3 with a preference for [conda](https://www.anaconda.com/)
- [CUDA](https://developer.nvidia.com/cuda-download) and a compatible [cuDNN](https://developer.nvidia.com/cudnn) installed system-wide

## Instructions

```powershell
git clone https://github.com/CBICA/GaNDLF.git
cd GaNDLF
conda create -p ./venv python=3.6 -y
conda activate ./venv
conda install requests pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y # install according to your cuda version https://pytorch.org/get-started/locally/
# conda install -c sdvillal openslide -y # this is required for windows
# conda install -c conda-forge libvips openslide -y # this is required for linux
pip install -e .
```
