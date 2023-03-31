# Setup/Installation Instructions

## Prerequisites

- Python3 with a preference for [conda](https://conda.io).
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
- Windows: [Microsoft Visual C++ 14.0 or greater](http://visualstudio.microsoft.com/visual-cpp-build-tools). This is required for PyTorch to work on Windows. If you are using conda, you can install it using the following command for your virtual environment: `conda install -c anaconda m2w64-toolchain`.

## Installation

### Install PyTorch 

GaNDLF's primary computational foundation is built on PyTorch, and as such it supports all hardware types that PyTorch supports. Please install PyTorch for your hardware type before installing GaNDLF. See the [PyTorch installation instructions](https://pytorch.org/get-started/locally) for more details. An example installation using CUDA, ROCm, and CPU-only is shown below:

```bash
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
```

### Optional Dependencies 

The following dependencies are optional, and are needed for specific features of GaNDLF.

```bash
(venv_gandlf) $> pip install openvino-dev==2022.1.0 # [OPTIONAL] to generate post-training optimized models for inference
(venv_gandlf) $> pip install mlcube_docker # [OPTIONAL] to deploy GaNDLF models as MLCube-compliant Docker containers
```

### Install from Package Managers

This option is recommended for most users, and allows for the quickest way to get started with GaNDLF.

```bash
# continue from previous shell
(venv_gandlf) $> pip install gandlf # this will give you the latest stable release
## you can also use conda
# (venv_gandlf) $> conda install -c conda-forge gandlf -y
```

If you are interested in running the latest version of GaNDLF, you can install the nightly build by running the following command:

```bash
# continue from previous shell
(venv_gandlf) $> pip install --pre gandlf
## you can also use conda
# (venv_gandlf) $> conda install -c conda-forge/label/gandlf_dev -c conda-forge gandlf -y
```


### Install from Sources

Use this option if you want to [contribute to GaNDLF](https://github.com/mlcommons/GaNDLF/blob/master/CONTRIBUTING.md), or are interested to make other code-level changes for your own use.

```bash
# continue from previous shell
(venv_gandlf) $> git clone https://github.com/mlcommons/GaNDLF.git
(venv_gandlf) $> cd GaNDLF
(venv_gandlf) $> pip install -e .
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
