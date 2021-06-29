# Setup/Installation Instructions

## Prerequisites

- Python3 with a preference for [conda](https://www.anaconda.com/)
- [CUDA](https://developer.nvidia.com/cuda-download) and a compatible [cuDNN](https://developer.nvidia.com/cudnn) installed system-wide

## Installation (WSL/ Linux)

```powershell
git clone https://github.com/CBICA/GaNDLF.git
cd GaNDLF
conda create -n venv_gandlf python=3.6 -y
conda activate venv_gandlf
conda install -c conda-forge mamba -y # allows for faster dependency solving
mamba install -c pytorch pytorch torchvision -y # 1.8.0 installs cuda 10.2 by default, personalize based on your cuda/driver availability via https://pytorch.org/get-started/locally/
mamba install -c conda-forge gandlf -y

# Or alternatively you can also use:
# conda install -c pytorch pytorch torchvision -y
# conda install -c conda-forge gandlf -y

## verify installation
python -c "import GANDLF as gf;print(gf.__version__)"
```

## Installation (Windows)
- Make sure you have [Microsoft Visual C++ 14.0 or greater](http://visualstudio.microsoft.com/visual-cpp-build-tools) installed.

```powershell
git clone https://github.com/CBICA/GaNDLF.git
cd GaNDLF
conda create -p ./venv python=3.7 -y
conda activate ./venv
conda install -c conda-forge mamba -y # allows for faster dependency solving
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch # customize based on your system cuda version appropriately
conda install -c sdvillal openslide -y
pip install -e .

## verify installation
python -c "import GANDLF as gf;print(gf.__version__)"
```
