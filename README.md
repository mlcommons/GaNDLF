# GANDLF

A **G**ener**A**lly **N**uanced **D**eep **L**earning **F**ramework for segmentation and classification.

## Other name candidates

- DeepSAGE: Deep SemAntic seGmEntator
- SEACAF: SEgmentation And ClassificAtion Framework
- DeepSAC: Deep Segmentation and Classification

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
- [Automatic mixed precision](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) support

## Documentation

Start at [./docs/index.md](./docs/index.md).

Includes:
- [Installation and setup](./docs/setup.md)
- [Using GANDLF]](./docs/usage.md)
- [Extending GANDLF]](./docs/extending.md)
