# deep-semantic-seg

## Installation

### Prerequisites

- Python3 with a preference for [conda](https://www.anaconda.com/)
- [CUDA](https://developer.nvidia.com/cuda-download) and a compatible [cuDNN](https://developer.nvidia.com/cudnn) installed system-wide

### Instructions

```powershell
conda create -p ./venv python=3.6.5 -y
conda activate ./venv
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -y # install according to your cuda version https://pytorch.org/get-started/locally/
pip install -e .
```

## To Do
- Single entry point for user (for both training and testing)
- Add more models that could potentially handle sparse data better
- Replace `batchgenerators` with `torchio` as it is easier to use and extend
- Put as many defaults as possible for different training/testing options in case the user passes bad argument in config file
- Consistent data I/O in a separate module so that this can be expanded for different tasks
- Single CSV (see train.csv)
  - read this into a pandas data frame
  - pass df to data set loader
  - parser the df on a per-header basis, keeping certain headers as "required", such as "Label" or "Mask"
- Use TorchIO to read this CSV
- Put CLI parameter parsing as a separate class for modularity and readability and this can be used by both the single interface for both training and testing