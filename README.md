# deep-semantic-seg

## Installation

```powershell
conda create -p ./venv python=3.6.5 -y
conda activate ./venv
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch # install according to your cuda version https://pytorch.org/get-started/locally/
pip install -e .
```

## To Do
- Input/output paths to be controlled by configuration file(s)
- Single entry point for user (for both training and testing)
- Add more models that could potentially handle sparse data better