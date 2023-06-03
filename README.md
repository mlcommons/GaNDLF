# GaNDLF

<p align="center">
  <a href="https://github.com/mlcommons/GaNDLF/actions/workflows/python-test.yml" alt="Build Status"><img src="https://github.com/mlcommons/GaNDLF/actions/workflows/python-test.yml/badge.svg" /></a>
  <a href="https://github.com/mlcommons/GaNDLF/actions/workflows/codeql-analysis.yml" alt="Code Analysis"><img src="https://github.com/mlcommons/GaNDLF/workflows/CodeQL/badge.svg" /></a>
  <a href="https://hub.docker.com/repository/docker/cbica/gandlf" alt="Docker CI"><img src="https://github.com/mlcommons/GaNDLF/actions/workflows/docker-image.yml/badge.svg" /></a>
  <a href="https://codecov.io/gh/mlcommons/GaNDLF" alt="Code Coverage"><img src="https://codecov.io/gh/mlcommons/GaNDLF/branch/master/graph/badge.svg?token=4I54XEI3WE" /></a>
  <a href="https://app.codacy.com/gh/mlcommons/GaNDLF?utm_source=github.com&utm_medium=referral&utm_content=mlcommons/GaNDLF&utm_campaign=Badge_Grade_Settings"><img alt="Codacy" src="https://api.codacy.com/project/badge/Grade/b2cf27ddce1b4907abb47a82931dcbca"></a><br>
  <a href="https://pypi.org/project/GANDLF/" alt="Install"><img src="https://img.shields.io/pypi/v/gandlf?color=blue" /></a>
  <a href="https://anaconda.org/conda-forge/gandlf" alt="Install"><img src="https://img.shields.io/conda/vn/conda-forge/gandlf?color=green" /></a>
  <a href="https://github.com/mlcommons/GaNDLF/discussions" alt="Issues"><img src="https://img.shields.io/badge/Support-Discussion-blue?color=red" /></a>
  <a href="https://doi.org/10.1038/s44172-023-00066-3" alt="Citation"><img src="https://img.shields.io/badge/Cite-citation-lightblue" /></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/Code%20Style-black-000000.svg"></a>
</p>

The **G**ener**a**lly **N**uanced **D**eep **L**earning **F**ramework for segmentation, regression and classification.

<p align="center">
    <img width="500" src="./docs/images/all_options_3.png" alt="GaNDLF all options">
</p>

## Why use this?

- Supports multiple
  - Deep Learning model architectures
  - Data dimensions (2D/3D)
  - Channels/images/sequences 
  - Prediction classes
  - Domain modalities (i.e., Radiology Scans and Digitized Histopathology Tissue Sections)
  - Problem types (segmentation, regression, classification)
  - Multi-GPU (on same machine) training
- Built-in 
  - Nested cross-validation (and related combined statistics)
  - Support for parallel HPC-based computing
  - Support for training check-pointing
  - Support for [Automatic mixed precision](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)
- Robust data augmentation, courtesy of [TorchIO](https://github.com/fepegar/torchio/)  
- Handles imbalanced classes (e.g., very small tumor in large organ)
- Leverages robust open source software
- No need to write any code to generate robust models

## Citation

Please cite the following article for GaNDLF ([full paper](https://www.nature.com/articles/s44172-023-00066-3)):

```bib
@article{pati2023gandlf,
    author={Pati, Sarthak and Thakur, Siddhesh P. and Hamamc{\i}, {\.{I}}brahim Ethem and Baid, Ujjwal and Baheti, Bhakti and Bhalerao, Megh and G{\"u}ley, Orhun and Mouchtaris, Sofia and Lang, David and Thermos, Spyridon and Gotkowski, Karol and Gonz{\'a}lez, Camila and Grenko, Caleb and Getka, Alexander and Edwards, Brandon and Sheller, Micah and Wu, Junwen and Karkada, Deepthi and Panchumarthy, Ravi and Ahluwalia, Vinayak and Zou, Chunrui and Bashyam, Vishnu and Li, Yuemeng and Haghighi, Babak and Chitalia, Rhea and Abousamra, Shahira and Kurc, Tahsin M. and Gastounioti, Aimilia and Er, Sezgin and Bergman, Mark and Saltz, Joel H. and Fan, Yong and Shah, Prashant and Mukhopadhyay, Anirban and Tsaftaris, Sotirios A. and Menze, Bjoern and Davatzikos, Christos and Kontos, Despina and Karargyris, Alexandros and Umeton, Renato and Mattson, Peter and Bakas, Spyridon},
    title={GaNDLF: the generally nuanced deep learning framework for scalable end-to-end clinical workflows},
    journal={Communications Engineering},
    year={2023},
    month={May},
    day={16},
    volume={2},
    number={1},
    pages={23},
    abstract={Deep Learning (DL) has the potential to optimize machine learning in both the scientific and clinical communities. However, greater expertise is required to develop DL algorithms, and the variability of implementations hinders their reproducibility, translation, and deployment. Here we present the community-driven Generally Nuanced Deep Learning Framework (GaNDLF), with the goal of lowering these barriers. GaNDLF makes the mechanism of DL development, training, and inference more stable, reproducible, interpretable, and scalable, without requiring an extensive technical background. GaNDLF aims to provide an end-to-end solution for all DL-related tasks in computational precision medicine. We demonstrate the ability of GaNDLF to analyze both radiology and histology images, with built-in support for k-fold cross-validation, data augmentation, multiple modalities and output classes. Our quantitative performance evaluation on numerous use cases, anatomies, and computational tasks supports GaNDLF as a robust application framework for deployment in clinical workflows.},
    issn={2731-3395},
    doi={10.1038/s44172-023-00066-3},
    url={https://doi.org/10.1038/s44172-023-00066-3}
}
```

## Documentation

GaNDLF has extensive documentation and it is arranged in the following manner:

- [Home](https://mlcommons.github.io/GaNDLF/)
- [Installation](https://mlcommons.github.io/GaNDLF/setup)
- [Usage](https://mlcommons.github.io/GaNDLF/usage)
- [Extension](https://mlcommons.github.io/GaNDLF/extending)
- [Frequently Asked Questions](https://mlcommons.github.io/GaNDLF/faq)
- [Acknowledgements](https://mlcommons.github.io/GaNDLF/acknowledgements)

## Contributing

Please see the [contributing guide](./CONTRIBUTING.md) for more information.

### Weekly Meeting

The GaNDLF development team hosts a weekly meeting to discuss feature additions, issues, and general future directions. If you are interested to join, please <a href="mailto:gandlf@mlcommons.org?subject=Meeting Request">send us an email</a>!

## Disclaimer
- The software has been designed for research purposes only and has neither been reviewed nor approved for clinical use by the Food and Drug Administration (FDA) or by any other federal/state agency.
- This code (excluding dependent libraries) is governed by [the Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt) provided in the [LICENSE file](./LICENSE) unless otherwise specified.

## Contact
For more information or any support, please post on the [Discussions](https://github.com/mlcommons/GaNDLF/discussions) section.
