# GaNDLF

A **G**ener**a**lly **N**uanced **D**eep **L**earning **F**ramework for segmentation and classification.

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

## Table of Contents

- [Application Setup](./setup.md)
- [Usage](./usage.md)
- [Extending GaNDLF](./extending.md)
