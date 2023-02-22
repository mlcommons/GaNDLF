This document will help you get started with GaNDLF using 3 representative examples using sample data:

- Segmentation
- Classification
- Regression

## Table of Contents


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
The contents of the `data` directory should look like this:

```bash
(venv_gandlf) $>  ls data
2d_histo_segmentation    2d_rad_segmentation    3d_rad_segmentation
# and a bunch of CSVs
```

**Note**: When using your own data, it is vital to correctly [prepare your data](https://mlcommons.github.io/GaNDLF/usage#preparing-the-data) prior to using it for any computational task (such as AI training or inference).

[Back To Top &uarr;](#table-of-contents)


## Segmentation using 3D Radiology Images

1. Download and extract the [sample data](#sample-data) as described above.
2. [Construct the main data file](https://mlcommons.github.io/GaNDLF/usage#constructing-the-data-csv) that will be used for the entire computation cycle. For the sample data, you can following these steps:

```python
# continue from the previous step
```