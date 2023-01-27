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

The instructions assume a system using NVIDIA GPUs with [CUDA 10.2](https://developer.nvidia.com/cuda-toolkit-archive) (for AMD, please make the appropriate change during PyTorch installation from [their installation page](https://pytorch.org/get-started/locally)).

```bash
git clone https://github.com/mlcommons/GaNDLF.git
cd GaNDLF
conda create -n venv_gandlf python=3.8 -y
conda activate venv_gandlf
### PyTorch LTS installation - https://pytorch.org/get-started/locally
## CUDA 11.3
# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
## CUDA 10.2
# pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102
## CPU-only
# pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install openvino-dev==2022.1.0 # [OPTIONAL] to generate optimized models for inference
pip install mlcube_docker # [OPTIONAL] to deploy GaNDLF models as MLCube-compliant Docker containers
pip install -e .

## alternatively you can also use:
# conda install -c pytorch pytorch torchvision -y
# conda install -c conda-forge gandlf -y

## verify installation
python ./gandlf_verifyInstall
```
