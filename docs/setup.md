# Setup/Installation Instructions

## Prerequisites

- Python3 with a preference for [conda](https://www.anaconda.com/).
- Knowledge of [managing Python environments](https://docs.python.org/3/tutorial/venv.html). The instructions below assume knowledge of the [conda management system](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

Alternatively, you can run GaNDLF via [Docker](https://www.docker.com/). This needs different prerequisites. See the [Docker Installation](#docker-installation) section below for more information. 

## Optional Requirements

- **GPU compute** (usually needed for faster training):
    - Install appropriate drivers:
        - [NVIDIA](https://www.nvidia.com/Download/index.aspx?lang=en-us)
        - [AMD](https://www.amd.com/en/support)
    - Compute toolkit appropriate for your hardware:
        - NVIDIA: [CUDA](https://developer.nvidia.com/cuda-download) and a compatible [cuDNN](https://developer.nvidia.com/cudnn) installed system-wide
        - AMD: [ROCm](https://www.amd.com/en/graphics/servers-solutions-rocm)
- Windows: [Microsoft Visual C++ 14.0 or greater](http://visualstudio.microsoft.com/visual-cpp-build-tools). This is required for PyTorch to work on Windows. If you are using conda, you can install it with `conda install -c anaconda m2w64-toolchain`.

## Installation

The instructions assume a system using NVIDIA GPUs with [CUDA 10.2](https://developer.nvidia.com/cuda-toolkit-archive) (for AMD, please make the appropriate change during PyTorch installation from [their installation page](https://pytorch.org/get-started/locally)).

```bash
(base) $> git clone https://github.com/mlcommons/GaNDLF.git
(base) $> cd GaNDLF
(base) $> conda create -n venv_gandlf python=3.8 -y
(base) $> conda activate venv_gandlf
(venv_gandlf) $> ### subsequent commands go here
### PyTorch installation - https://pytorch.org/get-started/locally
## CUDA 11.6
# (venv_gandlf) $> pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
## ROCm
# (venv_gandlf) $> pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2
## CPU-only
# (venv_gandlf) $> pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
(venv_gandlf) $> pip install openvino-dev==2022.1.0 # [OPTIONAL] to generate optimized models for inference
(venv_gandlf) $> pip install mlcube_docker # [OPTIONAL] to deploy GaNDLF models as MLCube-compliant Docker containers
(venv_gandlf) $> pip install -e .

## alternatively you can also use:
# (venv_gandlf) $> conda install -c pytorch pytorch torchvision -y
# (venv_gandlf) $> conda install -c conda-forge gandlf -y

## verify installation
(venv_gandlf) $> python ./gandlf_verifyInstall
```

Alternatively, GaNDLF can be installed via pip by running the following command:

```bash
# continue from previous shell
(venv_gandlf) $> pip install gandlf # this will give you the latest stable release
```

If you are interested in running the latest version of GaNDLF, you can install the nightly build by running the following command:

```bash
# continue from previous shell
(venv_gandlf) $> pip install --pre gandlf
```


## Docker Installation

We provide containerized versions of GaNDLF, which allows you to run GaNDLF without worrying about installation steps or dependencies.

### Steps to run the Docker version of GaNDLF

1. Install the [Docker Engine](https://www.docker.com/get-started) for your platform.
2. GaNDLF is available from [GitHub Package Registry](https://github.com/mlcommons/GaNDLF/pkgs/container/gandlf).
Several platform versions are available, including support for CUDA, ROCm, and CPU-only. Choose the one that best matches your system and drivers. For example, if you want to get the bleeding-edge GaNDLF version, and you have CUDA Toolkit v11.6, run the following command:

```bash
(base) $> docker pull ghcr.io/mlcommons/gandlf:latest-cuda116
```

This will download the GaNDLF image onto your machine. See the [usage page](https://mlcommons.github.io/GaNDLF/usage#running-with-docker) for details on how to run GaNDLF in this "dockerized" form.

### Enable GPU usage from Docker (optional, Linux only)

In order for "dockerized" GaNDLF to use your GPU, several steps are needed:

1. Ensure sure that you have correct NVIDIA drivers for your GPU.
2. Then, on Linux, follow the [instructions to set up the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit).
3. This can be replicated for ROCm for AMD , by following the [instructions to set up the ROCm Container Toolkit](https://rocmdocs.amd.com/en/latest/ROCm_Virtualization_Containers/ROCm-Virtualization-&-Containers.html?highlight=docker).

#### On Windows

On Windows, GPU and CUDA support requires either Windows 11, or (on Windows 10) to be registered for the Windows Insider program. If you meet those requirements and have [current NVIDIA drivers](https://developer.nvidia.com/cuda/wsl), GPU support for Docker [should work automatically](https://www.docker.com/blog/wsl-2-gpu-support-for-docker-desktop-on-nvidia-gpus/). Otherwise, please try updating your Docker Desktop version. 

**Note**: We cannot provide support for the Windows Insider program or for Docker Desktop itself.

### Building your own GaNDLF Docker Image

You may also build a Docker image of GaNDLF from the source repository. Just specify the `Dockerfile` for your preferred GPU-compute platform (or CPU):

```bash
(base) $> git clone https://github.com/mlcommons/GaNDLF.git
(base) $> cd GaNDLF
(base) $> docker build -t gandlf:${mytagname} -f Dockerfile-${target_platform} . # change ${mytagname} and ${target_platform} as needed
```
