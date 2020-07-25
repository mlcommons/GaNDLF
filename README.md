# deep-semantic-seg

## Name candidates

- DeepSAGE: Deep SemAntic seGmEntator

## Why use this?

- Supports multiple
  - Deep Learning model architectures
  - Channels/modalities 
  - Prediction classes
- Robust data augmentation, courtesy of [TorchIO](https://github.com/fepegar/torchio/)
- Built-in cross validation, with support for parallel HPC-based computing
- Leverages robust open source software
- No need to code to generate robust models

## Constructing the Data CSV

This application can leverage multiple channels/modalities for training while using a multi-class segmentation file. The expected format is shown as an example in [./configs/sample_train.csv](./configs/sample_train.csv) and needs to be structured with the following header format:

```
Channel_0,Channel_1,...Channel_X,Label
/full/path/0.nii.gz,/full/path/1.nii.gz,...,/full/path/X.nii.gz,/full/path/segmentation.nii.gz,
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

## To Do

- Generic multi-class segmentation support
- Ability to change [interpolation type](https://torchio.readthedocs.io/transforms/transforms.html?highlight=interpolation#interpolation) from config file
- Add option to normalize on a per-channel basis, if required
- Separate the training route into a separate function that takes the training + validation data and parameters as pickled objects from the main function
- Separate training code to make training more efficient for multi-fold training. Can possibly use https://schedule.readthedocs.io/en/stable/
- Single entry point for user (for both training and testing)
- Add more models that could potentially handle sparse data better
- Put as many defaults as possible for different training/testing options in case the user passes bad argument in config file
- Put CLI parameter parsing as a separate class for modularity and readability and this can be used by both the single interface for both training and testing
- Put downsampling as a parameter instead of hard-coding to 4
- Add option to train on multiple networks and then fuse results from all; basically some kind of ensemble
- Full-fledged preprocessing would be amazing
  - This would require additional dependencies, most notably CaPTk (which handles registration well and has a full suite of preprocessing tools)
    - Thinking more about this, it might _not_ be a great idea to do registration here, as it would be too opaque
    - Perhaps having a mechanism for intensity standardization (probably (Z-scoring)[https://torchio.readthedocs.io/transforms/preprocessing.html?highlight=intensity#torchio.transforms.ZNormalization] would be do)
  - Additional parameterization in the model configuration 
  - Sequence of operations are important
- [Model pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- Add appropriate architectures
  - from nnUnet
  - https://github.com/black0017/MedicalZooPytorch#implemented-architectures
- [Patch-based training](https://torchio.readthedocs.io/data/patch_training.html#patchsampler)
  - option for size in configuration file