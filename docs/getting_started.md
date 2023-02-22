This document will help you get started with GaNDLF using 3 representative examples using sample data:

- Segmentation
  - 3D radiology images
- Classification
- Regression

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Sample Data](#sample-data)
- [Segmentation using 3D Radiology Images](#segmentation-using-3d-radiology-images)
- [Classification using 2D Histology Images](#classification-using-2d-histology-images)


## Installation

Please follow the [installation instructions](./setup.md) to install GaNDLF. This should end up with the shell that looks like this with the GaNDLF virtual environment activated:

```bash
(venv_gandlf) $> ### subsequent commands goes here
```

[Back To Top &uarr;](#table-of-contents)


## Sample Data

We will use the sample data used for our extensive automated unit tests for all examples. The data is available from [this link](https://upenn.box.com/shared/static/y8162xkq1zz5555ye3pwadry2m2e39bs.zip):

```bash
(venv_gandlf) $> wget https://upenn.box.com/shared/static/y8162xkq1zz5555ye3pwadry2m2e39bs.zip -O ./gandlf_sample_data.zip
(venv_gandlf) $> unzip ./gandlf_sample_data.zip
# this should extract a directory called `data` in the current directory
```
The contents of the `data` directory should look like this (for brevity, this locations shall be referred to as `${GANDLF_DATA}` in the rest of the document):

```bash
(venv_gandlf) $>  ls data
2d_histo_segmentation    2d_rad_segmentation    3d_rad_segmentation
# and a bunch of CSVs which can be ignored
```

**Note**: When using your own data, it is vital to correctly [prepare your data](https://mlcommons.github.io/GaNDLF/usage#preparing-the-data) prior to using it for any computational task (such as AI training or inference).

[Back To Top &uarr;](#table-of-contents)


## Segmentation using 3D Radiology Images

1. Download and extract the [sample data](#sample-data) as described above.
2. [Construct the main data file](https://mlcommons.github.io/GaNDLF/usage#constructing-the-data-csv) that will be used for the entire computation cycle. For the sample data for this task, it should look like this:

    ```csv
    SubjectID,Channel_0,Label
    001,${GANDLF_DATA}/3d_rad_segmentation/001/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/001/mask.nii.gz
    002,${GANDLF_DATA}/3d_rad_segmentation/002/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/002/mask.nii.gz
    003,${GANDLF_DATA}/3d_rad_segmentation/003/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/003/mask.nii.gz
    004,${GANDLF_DATA}/3d_rad_segmentation/004/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/004/mask.nii.gz
    005,${GANDLF_DATA}/3d_rad_segmentation/005/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/005/mask.nii.gz
    006,${GANDLF_DATA}/3d_rad_segmentation/006/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/006/mask.nii.gz
    007,${GANDLF_DATA}/3d_rad_segmentation/007/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/007/mask.nii.gz
    008,${GANDLF_DATA}/3d_rad_segmentation/008/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/008/mask.nii.gz
    009,${GANDLF_DATA}/3d_rad_segmentation/009/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/009/mask.nii.gz
    010,${GANDLF_DATA}/3d_rad_segmentation/010/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/010/mask.nii.gz
    ```
3. [Construct the configuration file](https://mlcommons.github.io/GaNDLF/usage#customize-the-training) that will help design the computation (training and inference) pipeline. An example file for this task can be found [here](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_getting_started_segmentation_rad3d.yaml). This configuration has various levels of customization, and those details are [in this page](https://mlcommons.github.io/GaNDLF/customize.html).
4. Now you are ready to [train your model](https://mlcommons.github.io/GaNDLF/usage#running-gandlf-traininginference).
5. Once the model is trained, you can infer it on unseen data. Remember to construct a [similar data file](https://mlcommons.github.io/GaNDLF/usage#constructing-the-data-csv) for the unseen data, just without `Label` or `ValueToPredict` headers.

[Back To Top &uarr;](#table-of-contents)


## Classification using 2D Histology Images

1. Download and extract the [sample data](#sample-data) as described above.
2. [Extract patches/tiles from the full-size whole slide images](https://mlcommons.github.io/GaNDLF/usage#offline-patch-extraction-for-histology-images-only) for training. A sample configuration to extract patches is [here](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_getting_started_segmentation_rad2d_patchExtraction.yaml).
3. [Construct the main data file](https://mlcommons.github.io/GaNDLF/usage#constructing-the-data-csv) that will be used for the entire computation cycle. For the sample data for this task, it should get generated after [the patches are extracted](https://mlcommons.github.io/GaNDLF/usage#offline-patch-extraction-for-histology-images-only), and should look like this:

    ```csv
    SubjectID,Channel_0,Label
    001,${GANDLF_DATA}/3d_rad_segmentation/001/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/001/mask.nii.gz
    002,${GANDLF_DATA}/3d_rad_segmentation/002/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/002/mask.nii.gz
    003,${GANDLF_DATA}/3d_rad_segmentation/003/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/003/mask.nii.gz
    004,${GANDLF_DATA}/3d_rad_segmentation/004/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/004/mask.nii.gz
    005,${GANDLF_DATA}/3d_rad_segmentation/005/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/005/mask.nii.gz
    006,${GANDLF_DATA}/3d_rad_segmentation/006/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/006/mask.nii.gz
    007,${GANDLF_DATA}/3d_rad_segmentation/007/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/007/mask.nii.gz
    008,${GANDLF_DATA}/3d_rad_segmentation/008/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/008/mask.nii.gz
    009,${GANDLF_DATA}/3d_rad_segmentation/009/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/009/mask.nii.gz
    010,${GANDLF_DATA}/3d_rad_segmentation/010/image.nii.gz,${GANDLF_DATA}/3d_rad_segmentation/010/mask.nii.gz
    ```
4. [Construct the configuration file](https://mlcommons.github.io/GaNDLF/usage#customize-the-training) that will help design the computation (training and inference) pipeline. An example file for this task can be found [here](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_getting_started_segmentation_rad3d.yaml). This configuration has various levels of customization, and those details are [in this page](https://mlcommons.github.io/GaNDLF/customize.html).
5. Now you are ready to [train your model](https://mlcommons.github.io/GaNDLF/usage#running-gandlf-traininginference).
6. Once the model is trained, you can infer it on unseen data. Remember to construct a [similar data file](https://mlcommons.github.io/GaNDLF/usage#constructing-the-data-csv) for the unseen data, just without `Label` or `ValueToPredict` headers.

[Back To Top &uarr;](#table-of-contents)