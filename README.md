# deep-semantic-seg

## Name candidates

- DeepSAGE: Deep SemAntic seGmEntator
- SEACAF: SEgmentation And ClassificAtion Framework
- GANDLF (guess how to pronouce it): GenerAlly Nuanced Deep Learning Framework 

## Why use this?

- Supports multiple
  - Deep Learning model architectures
  - Channels/modalities 
  - Prediction classes
- Robust data augmentation, courtesy of [TorchIO](https://github.com/fepegar/torchio/)
- Built-in cross validation, with support for parallel HPC-based computing
- Multi-GPU (on the same machine) training
- Leverages robust open source software
- No need to code to generate robust models

## Constructing the Data CSV

This application can leverage multiple channels/modalities for training while using a multi-class segmentation file. The expected format is shown as an example in [./samples/sample_train.csv](./samples/sample_train.csv) and needs to be structured with the following header format:

```csv
Channel_0,Channel_1,...,Channel_X,Label
/full/path/0.nii.gz,/full/path/1.nii.gz,...,/full/path/X.nii.gz,/full/path/segmentation.nii.gz
```

- `Channel` can be substituted with `Modality` or `Image`
- `Label` can be substituted with `Mask` or `Segmentation`
- Only a single `Label` header should be passed (multiple segmentation classes should be in a single file with unique label numbers)

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

## Usage

```powershell
# continue from previous shell
python deepsage.py \
  -config ./experiment_0/model.cfg \ # model configuration
  -data ./experiment_0/train.csv \ # data in CSV format 
  -output ./experiment_0/output_dir/ \ # output directory
  -train 1 \ # 1 == train, 0 == inference
  -device 0 # postive integer for GPU device, -1 for CPU
  -modelDir /path/to/model/weights # used in inference mode
```

### Multi-GPU systems

Please ensure that the environment variable `CUDA_VISIBLE_DEVICES` is set [ref](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/).

## To Do

- Ability to change [interpolation type](https://torchio.readthedocs.io/transforms/transforms.html?highlight=interpolation#interpolation) from config file
- Add option to normalize on a per-channel basis, if required
- Multi-dimension architectures
- Single entry point for user 
  - Training: done
  - Inference
- Add more models that could potentially handle sparse data better
- Put as many defaults as possible for different training/testing options in case the user passes bad argument in config file
- Put CLI parameter parsing as a separate class for modularity and readability and this can be used by both the single interface for both training and testing
- Add option to train on multiple networks and then fuse results from all; basically some kind of ensemble
- [Model pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- Add appropriate architectures
  - from nnUnet
  - https://github.com/black0017/MedicalZooPytorch#implemented-architectures
- Regression example: https://github.com/wolny/pytorch-3dunet
- Ability to resume training if a compatible weight file is found in the output directory (how would this work for k-fold training)
- Ability to change the number of layers in the neural network models according to the parameter given in the congiguration file
- Change the way the training and inference is logged. More user friendly and less clutter
- Handling class imbalanaces
- Fix the learning rate schedule
- Different Augmentations for training and validation?
