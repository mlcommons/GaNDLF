# Setup/Installation Instructions

## Prerequisites

- Python3 with a preference for [conda](https://www.anaconda.com/)
- Knowledge of [managing python environments](https://docs.python.org/3/tutorial/venv.html); instructions below assume knowledge of the [conda management system](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- Windows: [Microsoft Visual C++ 14.0 or greater](http://visualstudio.microsoft.com/visual-cpp-build-tools)

Alternatively, you can run GaNDLF via [Docker](https://www.docker.com/). This needs different prerequisites. See the [Docker Installation](#docker-installation) section below for more information. 

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
### PyTorch installation - https://pytorch.org/get-started/locally
## CUDA 11.6
# pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
## ROCm
# pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2
## CPU-only
# pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install openvino-dev==2022.1.0 # [OPTIONAL] to generate optimized models for inference
pip install mlcube_docker # [OPTIONAL] to deploy GaNDLF models as MLCube-compliant Docker containers
pip install -e .

## alternatively you can also use:
# conda install -c pytorch pytorch torchvision -y
# conda install -c conda-forge gandlf -y

## verify installation
python ./gandlf_verifyInstall
```

## Docker Installation

We provide containerized versions of GaNDLF, allowing you to run GaNDLF without worrying about installation steps or dependencies.

First, install the Docker Engine by following the instructions for your platform at https://www.docker.com/get-started/.

GaNDLF is available from [GitHub Package Registry](https://github.com/mlcommons/GaNDLF/pkgs/container/gandlf).
Several platform versions are available, including CUDA 10.2, CUDA 11.3, and CPU-only. Choose the one that best matches your system and drivers.
For example, if you want to get the bleeding-edge GaNDLF version, and you have CUDA Toolkit v10.2, run the following command:

```bash
docker pull ghcr.io/mlcommons/gandlf:latest-cuda113
```

This will download the GaNDLF image onto your machine. See the [usage page](https://mlcommons.github.io/GaNDLF/usage#running-with-docker) for details on how to run GaNDLF in this "dockerized" form.

### Enable GPU usage from Docker (optional, Linux only)

In order for "dockerized" GaNDLF to use your GPU, several steps are needed. 

First, make sure that you have correct NVIDIA drivers for your GPU.

Then, on Linux, follow the [instructions to set up the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit).

#### On Windows

On Windows, GPU and CUDA support requires either Windows 11, or (on Windows 10) to be registered for the Windows Insider program.

If you meet those requirements and have [current NVIDIA drivers](https://developer.nvidia.com/cuda/wsl), GPU support for Docker [should work automatically](https://www.docker.com/blog/wsl-2-gpu-support-for-docker-desktop-on-nvidia-gpus/).
If it does not, please try updating your Docker Desktop version.
Please note that we cannot provide support for the Windows Insider program or for Docker Desktop itself.

### Building your own GaNDLF Docker Image

You may also build a Docker image of GaNDLF from the source repository.
Just specify the Dockerfile for your preferred GPU-compute platform (or CPU):

```bash
git clone https://github.com/mlcommons/GaNDLF.git
cd GaNDLF
docker build -t gandlf:myversion -f Dockerfile-CPU . # change to appropriate version of CUDA for the target platform
```


