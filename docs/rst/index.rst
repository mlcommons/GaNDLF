.. GANDLF documentation master file, created by
   sphinx-quickstart on Sun Aug 30 17:03:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GANDLF's documentation!
==================================

A **G**\ener-**A**\lly **N**\uanced **D**\eep **L**\earning **F**\ramework for segmentation and classification.

==================
Features
==================

- Supports multiple
  - Deep Learning model architectures
  - Channels/modalities 
  - Prediction classes
- Robust data augmentation, courtesy of [TorchIO](https://github.com/fepegar/torchio/)
- Built-in cross validation, with support for parallel HPC-based computing
- Multi-GPU (on the same machine) training
- Leverages robust open source software
- No need to code to generate robust models
- `Automatic Mixed Precision <https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/>`_ support


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   extending

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
