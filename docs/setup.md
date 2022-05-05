# Setup/Installation Instructions

## Prerequisites

- Python3 with a preference for [conda](https://www.anaconda.com/)
- Knowledge of [managing python environments](https://docs.python.org/3/tutorial/venv.html); instructions below assume knowledge of the [conda management system](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- Windows: [Microsoft Visual C++ 14.0 or greater](http://visualstudio.microsoft.com/visual-cpp-build-tools)

## Optional Requirements

- GPU compute: usually needed for faster training
  - Install appropriate drivers
    - [NVIDIA](https://www.nvidia.com/Download/index.aspx?lang=en-us)
    - [AMD](https://www.amd.com/en/support)
  - Compute toolkit appropriate for your hardware
    - NVIDIA: [CUDA](https://developer.nvidia.com/cuda-download) and a compatible [cuDNN](https://developer.nvidia.com/cudnn) installed system-wide
    - AMD: [ROCm](https://www.amd.com/en/graphics/servers-solutions-rocm)

## Installation

The instructions assume a system using NVIDA GPUs with [CUDA 10.2](https://developer.nvidia.com/cuda-toolkit-archive) (for AMD, please make the appropriate change during PyTorch installation from [their installation page](https://pytorch.org/get-started/locally)).

```bash
git clone https://github.com/CBICA/GaNDLF.git
cd GaNDLF
conda create -n venv_gandlf python=3.7 -y
conda activate venv_gandlf
### PyTorch LTS installation - https://pytorch.org/get-started/locally
## CUDA 10.2
# pip3 install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
## CUDA 11.1
# pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install openvino-dev==2022.1.0 # [OPTIONAL] to generate optimized models for inference
pip install -e .

## alternatively you can also use:
# conda install -c pytorch pytorch torchvision -y
# conda install -c conda-forge gandlf -y

## verify installation
python ./gandlf_verifyInstall
```
