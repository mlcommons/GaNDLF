# Usage

For any DL pipeline, the following flow needs to be performed:

1. Data preparation
2. Split data into training, validation, and testing
3. Customize the training parameters

GaNDLF tackles all of these and the details are split in the manner explained in [the following section](#table-of-contents).
## Table of Contents
- [Usage](#usage)
  - [Table of Contents](#table-of-contents)
  - [Preparing the Data](#preparing-the-data)
    - [Anonymize Data](#anonymize-data)
    - [Cleanup/Harmonize Data](#cleanupharmonize-data)
    - [Offline Patch Extraction (for histology images only)](#offline-patch-extraction-for-histology-images-only)
    - [Running preprocessing before training/inference](#running-preprocessing-before-traininginference)
  - [Constructing the Data CSV](#constructing-the-data-csv)
  - [Customize the Training](#customize-the-training)
  - [Running GaNDLF (Training/Inference)](#running-gandlf-traininginference)
  - [Plot the final results](#plot-the-final-results)
    - [Multi-GPU systems](#multi-gpu-systems)
  - [M3D-CAM usage](#m3d-cam-usage)
  - [Examples](#examples)

## Preparing the Data

### Anonymize Data

GaNDLF can anonymize single images or a collection of images using the `gandlf_anonymizer` script. The usage is as follows:
```bash
python gandlf_anonymizer
  # -h, --help         show help message and exit
  -c ./samples/config_anonymizer.yaml \ # anonymizer configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./input_dir_or_file \ # input directory containing series of images to anonymize or a single image
  -o ./output_dir_or_file # output directory to save anonymized images or a single output image file
```

### Cleanup/Harmonize Data

It is **highly** recommended that the dataset you want to train/infer on has been harmonized:

- Registration
  - Within-modality co-registration [[1](https://doi.org/10.1109/TMI.2014.2377694), [2](https://doi.org/10.1038/sdata.2017.117), [3](https://arxiv.org/abs/1811.02629)]
  - **OPTIONAL**: Registration of all datasets to patient atlas, if applicable [[1](https://doi.org/10.1109/TMI.2014.2377694), [2](https://doi.org/10.1038/sdata.2017.117), [3](https://arxiv.org/abs/1811.02629)]
- Size harmonization: Same physical definition of all images (see https://upenn.box.com/v/spacingsIssue for a presentation on how voxel resolutions affects downstream analyses). This is available via [GaNDLF's preprocessing module](#customize-the-training).
- Intensity harmonization: Same intensity profile, i.e., normalization [[4](https://doi.org/10.1016/j.nicl.2014.08.008), [5](https://visualstudiomagazine.com/articles/2020/08/04/ml-data-prep-normalization.aspx), [6](https://developers.google.com/machine-learning/data-prep/transform/normalization), [7](https://towardsdatascience.com/understand-data-normalization-in-machine-learning-8ff3062101f0)]. Z-scoring is available via [GaNDLF's preprocessing module](#customize-the-training).

Recommended tools for tackling all aforementioned preprocessing tasks: 
- [Cancer Imaging Phenomics Toolkit (CaPTk)](https://github.com/CBICA/CaPTk) 
- [Federated Tumor Segmentation (FeTS) Front End](https://github.com/FETS-AI/Front-End)

### Offline Patch Extraction (for histology images only)

GaNDLF can be used to convert a Whole Slide Image (WSI) with or without a corresponding label map to patches using [OPM](https://github.com/CBICA/OPM):

- Construct a YAML configuration for OPM with a minimum of the following keys (see [OPM usage](https://github.com/CBICA/OPM/blob/master/README.md#usage) for all options):
  - `scale`: scale at which operations such as tissue mask calculation happens; defaults to 16
  - `patch_size`: defines the size of the patches to extract, should be a tuple type of integers (e.g., [256,256])
  - `num_patches`: defines the number of patches to extract; use -1 to mine until exhaustion
- A CSV file with the following columns:
  - `SubjectID`: the ID of the subject for the WSI
  - `Channel_0`: the WSI file
  - `Label`: (optional) the label map file
- Run the following command:
```bash
python gandlf_patchMiner
  # -h, --help         show help message and exit
  -c ./exp_patchMiner/config.yaml \ # patch extraction configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./exp_patchMiner/input.csv \ # data in CSV format 
  -o ./exp_patchMiner/output_dir/ \ # output directory
```

### Running preprocessing before training/inference

This is optional, but recommended. It will significantly reduce the computational footprint during training/inference at the expense of larger storage requirements.
```bash
# continue from previous shell
python gandlf_preprocess \
  # -h, --help         show help message and exit
  -c ./experiment_0/model.yaml \ # model configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./experiment_0/train.csv \ # data in CSV format 
  -o ./experiment_0/output_dir/ \ # output directory
```

This will save the processed data in `./experiment_0/output_dir/` with a new data CSV and the corresponding model configuration.

[Back To Top &uarr;](#table-of-contents)


## Constructing the Data CSV

This application can leverage multiple channels/modalities for training while using a multi-class segmentation file. The expected format is shown as an example in [samples/sample_train.csv](https://github.com/CBICA/GaNDLF/blob/master/samples/sample_train.csv) and needs to be structured with the following header format (which shows a CSV with `N` subjects, each having `X` channels/modalities that need to be processed):

```csv
SubjectID,Channel_0,Channel_1,...,Channel_X,Label
001,/full/path/001/0.nii.gz,/full/path/001/1.nii.gz,...,/full/path/001/X.nii.gz,/full/path/001/segmentation.nii.gz
002,/full/path/002/0.nii.gz,/full/path/002/1.nii.gz,...,/full/path/002/X.nii.gz,/full/path/002/segmentation.nii.gz
...
N,/full/path/N/0.nii.gz,/full/path/N/1.nii.gz,...,/full/path/N/X.nii.gz,/full/path/N/segmentation.nii.gz
```

- `Channel` can be substituted with `Modality` or `Image`
- `Label` can be substituted with `Mask` or `Segmentation`and is used to specify the annotation file for segmentation models
- `ValueToPredict` is used for regression/classification models
- Only a single `Label` header should be passed (multiple segmentation classes should be in a single file with unique label numbers)

The [gandlf_constructCSV](https://github.com/CBICA/GaNDLF/blob/master/gandlf_constructCSV) can be used to make this easier:

```bash
# continue from previous shell
python gandlf_constructCSV \
  # -h, --help         show help message and exit
  -i ./experiment_0/data_dir/ # this is the main data directory
  -c _t1.nii.gz,_t1ce.nii.gz,_t2.nii.gz,_flair.nii.gz \ # 4 structural brain MR images
  -l _seg.nii.gz # label identifier - not needed for regression/classification
  -o ./experiment_0/train_data.csv \ # output CSV to be used for training
```

This assumes the data is in the following format:
```
./experiment_0/data_dir/
  │   │
  │   └───Patient_001 # this is used to construct the "SubjectID" header of the CSV
  │   │   │ Patient_001_brain_t1.nii.gz
  │   │   │ Patient_001_brain_t1ce.nii.gz
  │   │   │ Patient_001_brain_t2.nii.gz
  │   │   │ Patient_001_brain_flair.nii.gz
  │   │   │ Patient_001_brain_seg.nii.gz
  │   │   
  │   └───Patient_002 # this is used to construct the "Subject_ID" header of the CSV
  │   │   │ ...
  │
```

Notes:
- For classification/regression, add a column called `ValueToPredict`. Currently, we are supporting only a single value prediction per model.
- `SubjectID` or `PatientName` is used to ensure that the randomized split is done per-subject rather than per-image.

[Back To Top &uarr;](#table-of-contents)


## Customize the Training

GaNDLF requires a YAML-based configuration that controls various aspects of the training/inference process, such as:

- Model
  - Architecture
    - Segmentation: unet, resunet, uinc, fcn
    - Classification/Regression: 
      - DenseNet configurations: densenet121, densenet161, densenet169, densenet201, densenet264 
      - VGG configurations: vgg11, vgg13, vgg16, vgg19
  - Dimensionality of computations 
  - Final layer of model
  - Mixed precision
  - Class list
  - Model type: model used for inference, can be "torch" or "openvino"
      - "torch": use the learned Torch model for inference. If this parameter is not provided, it will default to be "Torch"
      - "openvino": use the OpenVINO Inference Engine for inference
- Various training parameters:
  - Patch size
  - Number of epochs and patience parameter
  - Learning rate
  - Scheduler 
  - Loss function
  - Optimizer
- Data Augmentation
- Data preprocessing
- Nested data splits
  - Testing 
  - Validation 

Please see a [sample](https://github.com/CBICA/GaNDLF/blob/master/samples/config_all_options.yaml) for detailed guide and comments.

- [Segmentation example](https://github.com/CBICA/GaNDLF/blob/master/samples/config_segmentation_brats.yaml)
- [Regression example](https://github.com/CBICA/GaNDLF/blob/master/samples/config_regression.yaml)
- [Classification example](https://github.com/CBICA/GaNDLF/blob/master/samples/config_classification.yaml)

**Note**: Ensure that the configuration has valid syntax by checking the file using any YAML validator such as https://yamlchecker.com/ or https://yamlvalidator.com/ **before** trying to train.

[Back To Top &uarr;](#table-of-contents)


## Running GaNDLF (Training/Inference)

```bash
# continue from previous shell
python gandlf_run \
  # -h, --help         show help message and exit
  # -v, --version      Show program's version number and exit.
  -c ./experiment_0/model.yaml \ # model configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./experiment_0/train.csv \ # data in CSV format 
  -m ./experiment_0/model_dir/ \ # model directory
  -t True \ # True == train, False == inference
  -d cuda # ensure CUDA_VISIBLE_DEVICES env variable is set for GPU device, use 'cpu' for CPU workloads
```

[Back To Top &uarr;](#table-of-contents)


## Plot the final results

After the testing/validation training is finished, GaNDLF makes it possible to collect all the statistics from the final models for testing and validation datasets and plot them. The [gandlf_collectStats](https://github.com/CBICA/GaNDLF/blob/master/gandlf_collectStats) can be used for this:

```bash
# continue from previous shell
python gandlf_collectStats \
  -m /path/to/trained/models \  # directory which contains testing and validation models
  -o ./experiment_0/output_dir_stats/  # output directory to save stats and plot
```

[Back To Top &uarr;](#table-of-contents)

### Multi-GPU systems

Please ensure that the environment variable `CUDA_VISIBLE_DEVICES` is set [[ref](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/)].

For an example how this is set, see [sge_wrapper](https://github.com/CBICA/GaNDLF/blob/master/samples/sge_wrapper).

[Back To Top &uarr;](#table-of-contents)


## M3D-CAM usage

The integration of the [M3D-CAM library](https://arxiv.org/abs/2007.00453) into GaNDLF enables the generation of attention maps for 3D/2D images in the validation epoch for classification and segmentation tasks.
To activate M3D-CAM one simply needs to add the following parameter to the config:

```yaml
medcam: 
{
  backend: "gcam",
  layer: "auto"
}
```

One can choose from the following backends:

- Grad-CAM (gcam)
- Guided Backpropagation (gbp)
- Guided Grad-CAM (ggcam)
- Grad-CAM++ (gcampp)

Optionally one can also change the name of the layer for which the attention maps should be generated.
The default behavior is "auto" which chooses the last convolutional layer.

All generated attention maps can be found in the experiment output_dir.
Link to the original repository: https://github.com/MECLabTUDA/M3d-Cam


## Examples

- Example data can be found in [the main repo](https://github.com/CBICA/GaNDLF/raw/master/testing/data.zip); this contains both 3D and 2D data that can be used to run various workloads.
- Configurations can be found in [the main repo](https://github.com/CBICA/GaNDLF/tree/master/testing).
