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
    issn={2731-3395},
    doi={10.1038/s44172-023-00066-3},
    url={https://doi.org/10.1038/s44172-023-00066-3}
}
```

## Contact
GaNDLF developers can be reached via the following ways:

- [GitHub Discussions](https://github.com/mlcommons/GaNDLF/discussions)
- [Email](mailto:gandlf@mlcommons.org)
