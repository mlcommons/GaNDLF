# GaNDLF

The **G**ener**a**lly **N**uanced **D**eep **L**earning **F**ramework (GaNDLF) for segmentation and classification.

## Why use GaNDLF?

- Supports multiple
  - Deep Learning model architectures
  - Channels/modalities 
  - Prediction classes
- Robust data augmentation, courtesy of [TorchIO](https://github.com/fepegar/torchio/) and [Albumentations](https://github.com/albumentations-team/albumentations)
- Built-in cross validation, with support for parallel HPC-based computing
- Multi-GPU (on the same machine) training
- Leverages robust open source software
- *Zero*-code needed to train robust models
- *Low*-code requirement for customization
- [Automatic mixed precision](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) support

## Table of Contents

- [Getting Started](./getting_started.md)
- [Application Setup](./setup.md)
- [Usage](./usage.md)
  - [Customize the training and inference](./customize.md)
- [Extending GaNDLF](./extending.md)
- [FAQ](./faq.md)
- [Acknowledgements](./acknowledgements.md)


## Contact
GaNDLF developers can be reached via the following ways:

- [GitHub Discussions](https://github.com/mlcommons/GaNDLF/discussions)
- [Email](mailto:gandlf@mlcommons.org)