# GaNDLF

<p align="center">
  <a href="https://github.com/CBICA/GaNDLF/actions/workflows/python-test.yml" alt="Build Status"><img src="https://github.com/CBICA/GaNDLF/actions/workflows/python-test.yml/badge.svg" /></a>
  <a href="https://github.com/CBICA/GaNDLF/actions/workflows/codeql-analysis.yml" alt="Code Analysis"><img src="https://github.com/CBICA/GaNDLF/workflows/CodeQL/badge.svg" /></a>
  <a href="https://hub.docker.com/repository/docker/cbica/gandlf" alt="Docker CI"><img src="https://github.com/CBICA/GaNDLF/actions/workflows/docker-image.yml/badge.svg" /></a>
  <a href="https://codecov.io/gh/CBICA/GaNDLF" alt="Code Coverage"><img src="https://codecov.io/gh/CBICA/GaNDLF/branch/master/graph/badge.svg?token=4I54XEI3WE" /></a>
  <a href="https://www.codacy.com/gh/CBICA/GaNDLF/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=CBICA/GaNDLF&amp;utm_campaign=Badge_Grade"><img alt="Codacy" src="https://app.codacy.com/project/badge/Grade/8f8b77f62ad843709534e4ed66ad0b5a"></a><br>
  <a href="https://anaconda.org/conda-forge/gandlf" alt="Install"><img src="https://img.shields.io/conda/vn/conda-forge/gandlf" /></a>
  <a href="https://github.com/CBICA/GaNDLF/discussions" alt="Issues"><img src="https://img.shields.io/badge/Support-Discussion-blue" /></a>
  <a href="https://arxiv.org/abs/2103.01006" alt="Citation"><img src="https://img.shields.io/badge/Cite-citation-lightblue" /></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/Code%20Style-black-000000.svg"></a>
</p>

The **G**ener**a**lly **N**uanced **D**eep **L**earning **F**ramework for segmentation, regression and classification.

<p align="center">
    <img width="500" src="./docs/images/all_options.png" alt="GaNDLF all options">
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

Please cite the following article for GaNDLF ([full PDF](https://arxiv.org/abs/2103.01006)):

```
@misc{pati2021gandlf,
      title={GaNDLF: A Generally Nuanced Deep Learning Framework for Scalable End-to-End Clinical Workflows in Medical Imaging}, 
      author={Sarthak Pati and Siddhesh P. Thakur and Megh Bhalerao and Spyridon Thermos and Ujjwal Baid and Karol Gotkowski and Camila Gonzalez and Orhun Guley and Ibrahim Ethem Hamamci and Sezgin Er and Caleb Grenko and Brandon Edwards and Micah Sheller and Jose Agraz and Bhakti Baheti and Vishnu Bashyam and Parth Sharma and Babak Haghighi and Aimilia Gastounioti and Mark Bergman and Anirban Mukhopadhyay and Sotirios A. Tsaftaris and Bjoern Menze and Despina Kontos and Christos Davatzikos and Spyridon Bakas},
      year={2021},
      eprint={2103.01006},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Documentation

GaNDLF has extensive documentation and it is arranged in the following manner:

- [Home](https://cbica.github.io/GaNDLF/)
- [Installation](https://cbica.github.io/GaNDLF/setup)
- [Usage](https://cbica.github.io/GaNDLF/usage)
- [Extension](https://cbica.github.io/GaNDLF/extending)
- [Frequently Asked Questions](https://cbica.github.io/GaNDLF/faq)
- [Acknowledgements](https://cbica.github.io/GaNDLF/acknowledgements)


## Contributing

Please see the [contributing guide](./CONTRIBUTING.md) for more information.

## Disclaimer
- The software has been designed for research purposes only and has neither been reviewed nor approved for clinical use by the Food and Drug Administration (FDA) or by any other federal/state agency.
- This code (excluding dependent libraries) is governed by the license provided in https://www.med.upenn.edu/cbica/software-agreement.html unless otherwise specified.

## Contact
For more information or any support, please post on the [Discussions](https://github.com/CBICA/GaNDLF/discussions) section or contact <a href="mailto:gandlf@cbica.upenn.edu">CBICA Software</a>.
