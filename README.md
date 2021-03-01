# GANDLF

A **G**ener**A**lly **N**uanced **D**eep **L**earning **F**ramework for segmentation, regression and classification.

## Why use this?

- Supports multiple
  - Deep Learning model architectures
  - Data dimensions (2D/3D)
  - Channels/modalities 
  - Prediction classes
  - Modalities (Radiology/Histopathology)
- Robust data augmentation, courtesy of [TorchIO](https://github.com/fepegar/torchio/)  
- Built-in nested cross validation (and related combined statistics), with support for parallel HPC-based computing
- Handles imbalanced classes (very small tumor in large organ)
- Multi-GPU (on the same machine) training
- Leverages robust open source software
- No need to code to generate robust models
- [Automatic mixed precision](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) support

## Citation

Publication pending on arXiv, due on 4th March 2021.

## Documentation

Start at [./docs/index.md](./docs/index.md).

Includes:
- [Installation and setup](./docs/setup.md)
- [Using GANDLF](./docs/usage.md)
- [Extending GANDLF](./docs/extending.md)
