## Introduction

For any DL pipeline, the following flow needs to be performed:

1. Data preparation
2. Split data into training, validation, and testing
3. Customize the training parameters

A detailed data flow diagram is presented in [this link](https://github.com/mlcommons/GaNDLF/blob/master/docs/README.md#flowchart).

GaNDLF addresses all of these, and the information is divided as described in the following sections.


## Installation

Please follow the [installation instructions](./setup.md#installation) to install GaNDLF. When the installation is complete, you should end up with the shell that looks like the following, which indicates that the GaNDLF virtual environment has been activated:

```bash
(venv_gandlf) $> ### subsequent commands go here
```


## Preparing the Data

### Anonymize Data

A major reason why one would want to anonymize data is to ensure that trained models do not inadvertently do not encode protect health information [[1](https://doi.org/10.1145/3436755),[2](https://doi.org/10.1038/s42256-020-0186-1)]. GaNDLF can anonymize single images or a collection of images using the `gandlf anonymizer` command. It can be used as follows:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf anonymizer
  # -h, --help         Show help message and exit
  -c ./samples/config_anonymizer.yaml \ # anonymizer configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./input_dir_or_file \ # input directory containing series of images to anonymize or a single image
  -o ./output_dir_or_file # output directory to save anonymized images or a single output image file (for a DICOM to NIfTi conversion specify a .nii.gz file)
```

### Cleanup/Harmonize/Curate Data

It is **highly** recommended that the dataset you want to train/infer on has been harmonized. The following requirements should be considered:

- Registration
    - Within-modality co-registration [[1](https://doi.org/10.1109/TMI.2014.2377694), [2](https://doi.org/10.1038/sdata.2017.117), [3](https://arxiv.org/abs/1811.02629)].
    - **OPTIONAL**: Registration of all datasets to patient atlas, if applicable [[1](https://doi.org/10.1109/TMI.2014.2377694), [2](https://doi.org/10.1038/sdata.2017.117), [3](https://arxiv.org/abs/1811.02629)].
- **Intensity harmonization**: Same intensity profile, i.e., normalization [[4](https://doi.org/10.1016/j.nicl.2014.08.008), [5](https://visualstudiomagazine.com/articles/2020/08/04/ml-data-prep-normalization.aspx), [6](https://developers.google.com/machine-learning/data-prep/transform/normalization), [7](https://towardsdatascience.com/understand-data-normalization-in-machine-learning-8ff3062101f0)]. GaNDLF offers [multiple options](#customize-the-training) for intensity normalization, including Z-scoring, Min-Max scaling, and Histogram matching. 
- **Resolution harmonization**: Ensures that the images have *similar* physical definitions (i.e., voxel/pixel size/resolution/spacing). An illustration of the impact of voxel size/resolution/spacing can be found [here](https://upenn.box.com/v/spacingsIssue), and it is encourage to read [this article](https://www.nature.com/articles/s41592-020-01008-z#:~:text=of%20all%20images.-,Resampling,-In%20some%20datasets) to added context on how this issue impacts a deep learning pipeline. This functionality is available via [GaNDLF's preprocessing module](#customize-the-training).

Recommended tools for tackling all aforementioned curation and annotation tasks: 
- [Cancer Imaging Phenomics Toolkit (CaPTk)](https://github.com/CBICA/CaPTk) 
- [Federated Tumor Segmentation (FeTS) Front End](https://github.com/FETS-AI/Front-End)
- [3D Slicer](https://www.slicer.org)
- [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php)

### Offline Patch Extraction (for histology images only)

GaNDLF can be used to convert a Whole Slide Image (WSI) with or without a corresponding label map to patches/tiles using GaNDLF’s integrated patch miner, which would need the following files:

1. A configuration file that dictates how the patches/tiles will be extracted. A sample configuration to extract patches is presented [here](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_getting_started_segmentation_histo2d_patchExtraction.yaml). The options that the can be defined in the configuration are as follows:
     - `patch_size`: defines the size of the patches to extract, should be a tuple type of integers (e.g., `[256,256]`) or a string containing patch size in microns (e.g., `[100m,100m]`). This parameter always needs to be specified.
     - `scale`: scale at which operations such as tissue mask calculation happens; defaults to `16`.
     - `num_patches`: defines the number of patches to extract, use `-1` to mine until exhaustion; defaults to `-1`.
     - `value_map`: mapping RGB values in label image to integer values for training; defaults to `None`.
     - `read_type`: either `random` or `sequential` (latter is more efficient); defaults to `random`.
     - `overlap_factor`: Portion of patches that are allowed to overlap (`0->1`); defaults to `0.0`.
     - `num_workers`: number of workers to use for patch extraction (note that this does not scale according to the number of threads available on your machine); defaults to `1`.
2. A CSV file with the following columns:
     - `SubjectID`: the ID of the subject for the WSI
     - `Channel_0`: the full path to the WSI file which will be used to extract patches
     - `Label`: (optional) full path to the label map file

Once these files are present, the patch miner can be run using the following command:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf patch-miner \ 
  # -h, --help         Show help message and exit
  -c ./exp_patchMiner/config.yaml \ # patch extraction configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./exp_patchMiner/input.csv \ # data in CSV format 
  -o ./exp_patchMiner/output_dir/ # output directory
```

### Running preprocessing before training/inference (optional)

Running preprocessing before training/inference is optional, but recommended. It will significantly reduce the computational footprint during training/inference at the expense of larger storage requirements. To run preprocessing before training/inference you can use the following command, which will save the processed data in `./experiment_0/output_dir/` with a new data CSV and the corresponding model configuration:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf preprocess \
  # -h, --help         Show help message and exit
  -c ./experiment_0/model.yaml \ # model configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./experiment_0/train.csv \ # data in CSV format 
  -o ./experiment_0/output_dir/ # output directory
```


## Constructing the Data CSV

This application can leverage multiple channels/modalities for training while using a multi-class segmentation file. The expected format is shown as an example in [samples/sample_train.csv](https://github.com/mlcommons/GaNDLF/blob/master/samples/sample_train.csv) and needs to be structured with the following header format (which shows a CSV with `N` subjects, each having `X` channels/modalities that need to be processed):

```csv
SubjectID,Channel_0,Channel_1,...,Channel_X,Label
001,/full/path/001/0.nii.gz,/full/path/001/1.nii.gz,...,/full/path/001/X.nii.gz,/full/path/001/segmentation.nii.gz
002,/full/path/002/0.nii.gz,/full/path/002/1.nii.gz,...,/full/path/002/X.nii.gz,/full/path/002/segmentation.nii.gz
...
N,/full/path/N/0.nii.gz,/full/path/N/1.nii.gz,...,/full/path/N/X.nii.gz,/full/path/N/segmentation.nii.gz
```

**Notes:**

- `Channel` can be substituted with `Modality` or `Image`
- `Label` can be substituted with `Mask` or `Segmentation`and is used to specify the annotation file for segmentation models
- For classification/regression, add a column called `ValueToPredict`. Currently, we are supporting only a single value prediction per model.
- Only a single `Label` or `ValueToPredict` header should be passed 
    - Multiple segmentation classes should be in a single file with unique label numbers.
    - Multi-label classification/regression is currently not supported.

### Using the `gandlf construct-csv` command

To make the process of creating the CSV easier, we have provided a `gandlf construct-csv` command. This script works when the data is arranged in the following format (example shown of the data directory arrangement from the [Brain Tumor Segmentation (BraTS) Challenge](https://www.synapse.org/brats)):

```bash
$DATA_DIRECTORY
│
└───Patient_001 # this is constructed from the ${PatientID} header of CSV
│   │ Patient_001_brain_t1.nii.gz
│   │ Patient_001_brain_t1ce.nii.gz
│   │ Patient_001_brain_t2.nii.gz
│   │ Patient_001_brain_flair.nii.gz
│   │ Patient_001_seg.nii.gz # optional for segmentation tasks
│
└───Patient_002 # this is constructed from the ${PatientID} header of CSV
│   │ Patient_002_brain_t1.nii.gz
│   │ Patient_002_brain_t1ce.nii.gz
│   │ Patient_002_brain_t2.nii.gz
│   │ Patient_002_brain_flair.nii.gz
│   │ Patient_002_seg.nii.gz # optional for segmentation tasks
│
└───JaneDoe # this is constructed from the ${PatientID} header of CSV
│   │ randomFileName_0_t1.nii.gz # the string identifier needs to be the same for each modality
│   │ randomFileName_1_t1ce.nii.gz
│   │ randomFileName_2_t2.nii.gz
│   │ randomFileName_3_flair.nii.gz
│   │ randomFileName_seg.nii.gz # optional for segmentation tasks
│
...
```

The following command shows how the script works:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf construct-csv \
  # -h, --help         Show help message and exit
  -i $DATA_DIRECTORY # this is the main data directory 
  -c _t1.nii.gz,_t1ce.nii.gz,_t2.nii.gz,_flair.nii.gz \ # an example image identifier for 4 structural brain MR sequences for BraTS, and can be changed based on your data
  -l _seg.nii.gz \ # an example label identifier - not needed for regression/classification, and can be changed based on your data
  -o ./experiment_0/train_data.csv # output CSV to be used for training
```

**Notes**:

- For classification/regression, add a column called `ValueToPredict`. Currently, we are supporting only a single value prediction per model.
- `SubjectID` or `PatientName` is used to ensure that the randomized split is done per-subject rather than per-image.
- For data arrangement different to what is described above, a customized script will need to be written to generate the CSV, or you can enter the data manually into the CSV. 

### Using the `gandlf split-csv` command

To split the data CSV into training, validation, and testing CSVs, the `gandlf split-csv` script can be used. The following command shows how the script works:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf split-csv \
  # -h, --help         Show help message and exit
  -i ./experiment_0/train_data.csv \ # output CSV from the `gandlf construct-csv` script
  -c $gandlf_config \ # the GaNDLF config (in YAML) with the `nested_training` key specified to the folds needed
  -o $output_dir # the output directory to save the split data
```

### Using the `--log-file` parameter
By default, only the `info` and `error` logs will be **displayed** in the console and
the log file will be **saved** in `$(home)/.gandlf/<timestamp>.log`.

Also, you can use the `--log-file` and provide the file that you want to save the logs
```bash
(venv_gandlf) $> gandlf <command> --log-file <log_file_path>
```

## Customize the Training

GaNDLF requires a YAML-based configuration that controls various aspects of the training/inference process. There are multiple samples for users to start as their baseline for further customization. A list of the available samples is presented as follows:

- [Sample showing all the available options](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_all_options.yaml)
- [Segmentation example](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_segmentation_brats.yaml)
- [Regression example](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_regression.yaml)
- [Classification example](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_classification.yaml)

**Notes**: 

- More details on the configuration options are available in the [customization page](customize.md).
- Ensure that the configuration has valid syntax by checking the file using any YAML validator such as [yamlchecker.com](https://yamlchecker.com/) or [yamlvalidator.com](https://yamlvalidator.com/) **before** trying to train.

### Running multiple experiments (optional)

1. The `gandlf config-generator` command can be used to generate a grid of configurations for tuning the hyperparameters of a baseline configuration that works for your dataset and problem. 
2. Use a strategy file (example is shown in [samples/config_generator_strategy.yaml](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_generator_sample_strategy.yaml).
3. Provide the baseline configuration which has enabled you to successfully train a model for `1` epoch for your dataset and problem at hand (regardless of the efficacy).
4. Run the following command:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf config-generator \
  # -h, --help         Show help message and exit
  -c ./samples/config_all_options.yaml \ # baseline configuration
  -s ./samples/config_generator_strategy.yaml \ # strategy file
  -o ./all_experiments/ # output directory
```
5. For example, to generate `4` configurations that leverage `unet` and `resunet` architectures for learning rates of `[0.1,0.01]`,  you can use the following strategy file:
```yaml
model:
  {
    architecture: [unet, resunet],
  }
learning_rate: [0.1, 0.01]
```


## Running GaNDLF (Training/Inference)

You can use the following code snippet to run GaNDLF:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf run \
  # -h, --help         Show help message and exit
  # -v, --version      Show program's version number and exit.
  -c ./experiment_0/model.yaml \ # model configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./experiment_0/train.csv \ # data in CSV format 
  -m ./experiment_0/model_dir/ \ # model directory (i.e., the `model-dir`) where the output of the training will be stored, created if not present 
  --train \ # --train/-t or --infer
  # ensure CUDA_VISIBLE_DEVICES env variable is set when using GPU 
  # -rt , --reset # [optional] completely resets the previous run by deleting `model-dir`
  # -rm , --resume # [optional] resume previous training by only keeping model dict in `model-dir`
```

### Special notes for Inference for Histology images

- If you trying to perform inference on pre-extracted patches, please change the `modality` key in the configuration to `rad`. This will ensure the histology-specific pipelines are not triggered.
- However, if you are trying to perform inference on full WSIs, `modality` should be kept as `histo`.


## Generate Metrics 

GaNDLF provides a script to generate metrics after an inference process is done.The script can be used as follows:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf generate-metrics \
  # -h, --help         Show help message and exit
  # -v, --version      Show program's version number and exit.
  -c , --config       The configuration file (contains all the information related to the training/inference session)
  -i , --input-data    CSV file that is used to generate the metrics; should contain 3 columns: 'SubjectID,Target,Prediction'
  -o , --output-file   Location to save the output dictionary. If not provided, will print to stdout.
```

Once you have your CSV in the specific format, you can pass it on to generate the metrics. Here is an example for segmentation:

```csv
SubjectID,Target,Prediction
001,/path/to/001/target.nii.gz,/path/to/001/prediction.nii.gz
002,/path/to/002/target.nii.gz,/path/to/002/prediction.nii.gz
...
```

Similarly, for classification or regression (`A`, `B`, `C`, `D` are integers for classification and floats for regression):

```csv
SubjectID,Target,Prediction
001,A,B
002,C,D
...
```

### Special cases 

- **BraTS Segmentation Metrics**: To generate annotation to annotation metrics for BraTS segmentation tasks [[ref](https://www.synapse.org/brats)], ensure that the config has `problem_type: segmentation_brats`, and the CSV can be in the same format as segmentation:

  ```csv
  SubjectID,Target,Prediction
  001,/path/to/001/target_image.nii.gz,/path/to/001/prediction_image.nii.gz
  002,/path/to/002/target_image.nii.gz,/path/to/002/prediction_image.nii.gz
  ...
  ```

  This can also be customized using the `panoptica_config` dictionary. See [this sample](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_segmentation_metrics_brats_default.yaml) for an example. Additionally, a more "concise" variant of the config is present [here](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_segmentation_metrics_brats_concise.yaml).

- **BraTS Synthesis Metrics**: To generate image to image metrics for synthesis tasks (including for the BraTS synthesis tasks [[1](https://www.synapse.org/#!Synapse:syn51156910/wiki/622356), [2](https://www.synapse.org/#!Synapse:syn51156910/wiki/622357)]), ensure that the config has `problem_type: synthesis`, and the CSV can be in the same format as segmentation (note that the `Mask` column is optional):

  ```csv
  SubjectID,Target,Prediction,Mask
  001,/path/to/001/target_image.nii.gz,/path/to/001/prediction_image.nii.gz,/path/to/001/brain_mask.nii.gz
  002,/path/to/002/target_image.nii.gz,/path/to/002/prediction_image.nii.gz,/path/to/002/brain_mask.nii.gz
  ...
  ```


## GPU usage, parallelization and numerical precision

### Using GPU
By default, GaNDLF will use GPU accelerator if `CUDA_VISIBLE_DEVICES` environment variable is set [ref](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/). To explicitly configure usage of GPU/CPU, you can set the `accelerator` key in the configuration file to `cuda` or `cpu`. By default, it is set to `auto`, which will use GPU if available, otherwise CPU.

### Multi-GPU training

GaNDLF enables relatively straightforward multi-GPU training. Simply set the `CUDA_VISIBLE_DEVICES` environment variable includes the GPU IDs you want to use (see above paragraph for reference about this variable). GaNDLF will automatically distribute the training across the GPUs using Distributed Data Parallel (DDP) strategy. To explicitly set the strategy, set the `strategy` key in the configuration file to `ddp`. By default, it is set to `auto`, which will use DDP if multiple GPUs are available, otherwise single GPU or CPU.

### Multi-node training and HPC environments
In the current release, GaNDLF does not support multi-node training. This feature will be enabled in the upcoming releases.

### Choosing numerical precision
By default, GaNDLF uses `float32` for training and inference computations. To change the numerical precision, set the `precision` key in the configuration file to one of the values `[16, 32, 64, "64-true", "32-true", "16-mixed", "bf16", "bf16-mixed"]`. Using reduced precision usually results in faster computations and lower memory usage, although in some cases it may lead to numerical instability - be sure to observe the training process and adjust the precision accordingly. Good rule of thumb is to start in default precision (`32`) and then experiment with lower precisions.

## Expected Output(s)

### Training

Once your model is trained, you should see the following output:

```bash
# continue from previous shell
(venv_gandlf) $> ls ./experiment_0/model_dir/
data_${cohort_type}.csv  # data CSV used for the different cohorts, which can be either training/validation/testing
data_${cohort_type}.pkl  # same as above, but in pickle format
logs_${cohort_type}.csv  # logs for the different cohorts that contain the various metrics, which can be either training/validation/testing
${architecture_name}_best.pth.tar # the best model in native PyTorch format
${architecture_name}_latest.pth.tar # the latest model in native PyTorch format
${architecture_name}_initial.pth.tar # the initial model in native PyTorch format
${architecture_name}_initial.{onnx/xml/bin} # [optional] if ${architecture_name} is supported, the graph-optimized best model in ONNX format
# other files dependent on if training/validation/testing output was enabled in configuration
```

### Inference

- The output of inference will be predictions based on the model that was trained. 
- The predictions will be saved in the same directory as the model if `output-dir` is not passed to `gandlf run`.
- For segmentation, a directory will be created per subject ID in the input CSV.
- For classification/regression, the predictions will be generated in the `output-dir` or `model-dir` as a CSV file.


## Plot the final results

After the testing/validation training is finished, GaNDLF enables the collection of all the statistics from the final models for testing and validation datasets and plot them. The [gandlf collect-stats](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/entrypoints/collect_stats.py) command can be used for plotting:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf collect-stats \
  -m /path/to/trained/models \  # directory which contains testing and validation models
  -o ./experiment_0/output_dir_stats/  # output directory to save stats and plot
```


## M3D-CAM usage

The integration of the [M3D-CAM library](https://arxiv.org/abs/2007.00453) into GaNDLF enables the generation of attention maps for 3D/2D images in the validation epoch for classification and segmentation tasks.
To activate M3D-CAM you just need to add the following parameter to the config:

```yaml
medcam: 
{
  backend: "gcam",
  layer: "auto"
}
```

You can choose one of the following backends:

- Grad-CAM (`gcam`)
- Guided Backpropagation (`gbp`)
- Guided Grad-CAM (`ggcam`)
- Grad-CAM++ (`gcampp`)

Optionally one can also change the name of the layer for which the attention maps should be generated. The default behavior is `auto` which chooses the last convolutional layer.

All generated attention maps can be found in the experiment's output directory. Link to the original repository: [github.com/MECLabTUDA/M3d-Cam](https://github.com/MECLabTUDA/M3d-Cam)


## Post-Training Model Optimization

If you have a model previously trained using GaNDLF that you wish to run graph optimizations on, you can use the `gandlf optimize-model` command to do so. The following command shows how it works:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf optimize-model \
  -m /path/to/trained/${architecture_name}_best.pth.tar  # directory which contains testing and validation models
```

If `${architecture_name}` is supported, the optimized model will get generated in the model directory, with the name `${architecture_name}_optimized.onnx`.


## Deployment

### Deploy as a Model (WARNING - this feature is not fully supported in the current release!)

GaNDLF provides the ability to deploy models into easy-to-share, easy-to-use formats -- users of your model do not even need to install GaNDLF. Currently, Docker images are supported (which can be converted to [Apptainer/Singularity format](https://apptainer.org/docs/user/main/docker_and_oci.html)). These images meet [the MLCube interface](https://mlcommons.org/en/mlcube/). This allows your algorithm to be used in a consistent manner with other machine learning tools.

The resulting image contains your specific version of GaNDLF (including any custom changes you have made) and your trained model and configuration. This ensures that upstream changes to GaNDLF will not break compatibility with your model.

To deploy a model, simply run the `gandlf deploy` command after training a model. You will need the [Docker engine](https://www.docker.com/get-started/) installed to build Docker images. This will create the image and, for MLCubes, generate an MLCube directory complete with an `mlcube.yaml` specifications file, along with the workspace directory copied from a pre-existing template. 

```bash
# continue from previous shell
(venv_gandlf) $> gandlf deploy \
  # -h, --help         Show help message and exit
  -c ./experiment_0/model.yaml \ # Configuration to bundle with the model (you can recover it with `gandlf recover-config` first if needed)
  -m ./experiment_0/model_dir/ \ # model directory (i.e., modeldir)
  --target docker \ # the target platform (--help will show all available targets)
  --mlcube-root ./my_new_mlcube_dir \ # Directory containing mlcube.yaml (used to configure your image base)
  -o ./output_dir # Output directory where a  new mlcube.yaml file to be distributed with your image will be created
  --mlcube-type model # deploy as a model MLCube.
```

### Deploy as a Metrics Generator

You can also deploy GaNDLF as a metrics generator (see the [Generate Metrics](#generate-metrics) section) as follows:

```bash
(venv_gandlf) $> gandlf deploy \
  ## -h, --help         show help message and exit
  --target docker \ # the target platform (--help will show all available targets)
  --mlcube-root ./my_new_mlcube_dir \ # Directory containing mlcube.yaml (used to configure your image base)
  -o ./output_dir # Output directory where a  new mlcube.yaml file to be distributed with your image will be created
  -e ./my_custom_script.py # An optional custom script used as an entrypoint for your MLCube
  --mlcube-type metrics # deploy as a metrics MLCube.
```

The resulting MLCube can be used to calculate any metrics supported in GaNDLF. You can configure which metrics to be calculated by passing a GaNDLF config file when running the MLCube.

For more information about using a custom entrypoint script, see the examples [here](https://github.com/mlcommons/GaNDLF/tree/master/mlcube).

## Federating your model using OpenFL

Once you have a model definition, it is easy to perform federated learning using the [Open Federated Learning (OpenFL) library](https://github.com/securefederatedai/openfl). Follow the tutorial in [this page](https://openfl.readthedocs.io/en/latest/developer_guide/running_the_federation_with_gandlf.html) to get started.


## Federating your model evaluation using MedPerf

Once you have a trained model, you can perform [federated evaluation](https://flower.dev/docs/tutorial/Flower-0-What-is-FL.html#Federated-evaluation) using [MedPerf](https://medperf.org/). Follow the tutorial in [this page](https://docs.medperf.org/mlcubes/gandlf_mlcube/) to get started.

**Notes**:
-  To create a GaNDLF MLCube, for technical reasons, you need write access to the GaNDLF package. This should be automatic while using a virtual environment that you have set up. See the [installation instructions](./setup.md#installation) for details.
-  This needs GaNDLF to be initialized as an MLCube. See [the mlcube instructions](https://docs.medperf.org/mlcubes/gandlf_mlcube/) for details.

## Running with Docker

The usage of GaNDLF remains generally the same even from Docker, but there are a few extra considerations.

Once you have pulled the GaNDLF image, it will have a tag, such as `cbica/gandlf:latest-cpu`. Run the following command to list your images and ensure GaNDLF is present:

```bash
(main) $> docker image ls
```

You can invoke `docker run` with the appropriate tag to run GaNDLF:

```bash
(main) $> docker run -it --rm --name gandlf cbica/gandlf:latest-cpu ${gandlf command and parameters go here!}
```

Remember that arguments/options for *Docker itself* go *before* the image tag, while the command and arguments for GaNDLF go *after* the image tag. For more details and options, see the [Docker run documentation](https://docs.docker.com/engine/reference/commandline/run/).

However, most commands that require files or directories as input or output will fail, because the container, by default, cannot read or write files on your machine for [security considerations](https://docs.docker.com/develop/security-best-practices/). In order to fix this, you need to [mount specific locations in the filesystem](#mounting-input-and-output). 

### Mounting Input and Output

The container is basically a filesystem of its own. To make your data available to the container, you will need to mount in files and directories. Generally, it is useful to mount at least input directory (as read-only) and an output directory. See the [Docker bind mount instructions](https://docs.docker.com/storage/bind-mounts/) for more information.

For example, you might run:

```bash
(main) $> docker run -it --rm --name gandlf --volume /home/researcher/gandlf_input:/input:ro --volume /home/researcher/gandlf_output:/output cbica/gandlf:latest-cpu [command and args go here]
```

Remember that the process running in the container only considers the filesystem inside the container, which is structured differently from that of your host machine. Therefore, you will need to give paths relative to the mount point *destination*. Additionally, any paths used internally by GaNDLF will refer to locations inside the container. This means that data CSVs produced by the `gandlf construct-csv` command will need to be made from the container and with input in the same locations. Expanding on our last example:

```bash
(main) $> docker run -it --rm --name dataprep \
  --volume /home/researcher/gandlf_input:/input:ro \ # input data is mounted as read-only
  --volume /home/researcher/gandlf_output:/output \ # output data is mounted as read-write
  cbica/gandlf:latest-cpu \ # change to appropriate docker image tag
  construct-csv \ # standard construct CSV API starts
  --input-dir /input/data \
  --output-file /output/data.csv \
  --channels-id _t1.nii.gz \
  --label-id _seg.nii.gz
```

The previous command will generate a data CSV file that you can safely edit outside the container (such as by adding a `ValueToPredict` column). Then, you can refer to the same file when running again:

```bash
(main) $> docker run -it --rm --name training \
  --volume /home/researcher/gandlf_input:/input:ro \ # input data is mounted as read-only
  --volume /home/researcher/gandlf_output:/output \ # output data is mounted as read-write
  cbica/gandlf:latest-cpu \ # change to appropriate docker image tag
  gandlf run --train \ # standard training API starts
  --config /input/config.yml \
  --inputdata /output/data.csv \
  --modeldir /output/model
```
#### Special Case for Training

Considering that you want to train on an existing model that is inside the GaNDLF container (such as in an MLCube container created by `gandlf deploy`), the output will be to a location embedded inside the container. Since you cannot mount something into that spot without overwriting the model, you can instead use the built-in `docker cp` command to extract the model afterward. For example, you can fine-tune a model on your own data using the following commands as a starting point:

```bash
# Run training on your new data
(main) $> docker run --name gandlf_training mlcommons/gandlf-pretrained:0.0.1 -v /my/input/data:/input gandlf run -m /embedded_model/ [...] # Do not include "--rm" option!
# Copy the finetuned model out of the container, to a location on the host
(main) $> docker cp gandlf_training:/embedded_model /home/researcher/extracted_model
# Now you can remove the container to clean up
(main) $> docker rm -f gandlf_training
```

### Enabling GPUs

Some special arguments need to be passed to Docker to enable it to use your GPU. With Docker version > 19.03 You can use `docker run --gpus all` to expose all GPUs to the container. See the [NVIDIA Docker documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration) for more details.

If using CUDA, GaNDLF also expects the environment variable `CUDA_VISIBLE_DEVICES` to be set. To use the same settings as your host machine, simply add `-e CUDA_VISIBLE_DEVICES` to your docker run command. For example:

For example:
```bash
(main) $> docker run --gpus all -e CUDA_VISIBLE_DEVICES -it --rm --name gandlf cbica/gandlf:latest-cuda113 gandlf run --device cuda [...]
```

This can be replicated for ROCm for AMD , by following the [instructions to set up the ROCm Container Toolkit](https://rocmdocs.amd.com/en/latest/ROCm_Virtualization_Containers/ROCm-Virtualization-&-Containers.html?highlight=docker).

## MLCubes

GaNDLF, and GaNDLF-created models, may be distributed as an [MLCube](https://mlcommons.github.io/mlcube/). This involves distributing an `mlcube.yaml` file. That file can be specified when using the [MLCube runners](https://mlcommons.github.io/mlcube/runners/). The runner will perform many aspects of configuring your container for you. Currently, only the `mlcube_docker` runner is supported. 

See the [MLCube documentation](https://mlcommons.github.io/mlcube/) for more details.

## HuggingFace CLI

This tool allows you to interact with the Hugging Face Hub directly from a terminal. For example, you can create a repository, upload and download files, etc.

### Download an entire repository
GaNDLF's Hugging Face CLI allows you to download repositories through the command line. This can be done by just specifying the repo id:

```bash
(main) $> gandlf hf --download --repo-id HuggingFaceH4/zephyr-7b-beta
```

Apart from the Repo Id you can also provide other arguments.

### Revision
To download from a specific revision (commit hash, branch name or tag), use the --revision option:
```bash
(main) $> gandlf hf --download --repo-id distilbert-base-uncased revision --revision v1.1
```
### Specify a token
To access private or gated repositories, you must use a token. You can do this using the --token option:

```bash
(main) $> gandlf hf --download --repo-id distilbert-base-uncased revision --revision v1.1 --token hf_****
```

### Specify cache directory
If not using --local-dir, all files will be downloaded by default to the cache directory defined by the HF_HOME environment variable. You can specify a custom cache using --cache-dir:

```bash
(main) $> gandlf hf --download --repo-id distilbert-base-uncased revision --revision v1.1 --token hf_**** --cache-dir ./path/to/cache
```

### Download to a local folder
The recommended (and default) way to download files from the Hub is to use the cache-system. However, in some cases you want to download files and move them to a specific folder. This is useful to get a workflow closer to what git commands offer. You can do that using the --local-dir option.

A ./huggingface/ folder is created at the root of your local directory containing metadata about the downloaded files. This prevents re-downloading files if they’re already up-to-date. If the metadata has changed, then the new file version is downloaded. This makes the local-dir optimized for pulling only the latest changes.

```bash
(main) $> gandlf hf --download --repo-id distilbert-base-uncased revision --revision v1.1 --token hf_**** --cache-dir ./path/to/cache --local-dir ./path/to/dir
```
### Force Download
To specify if the files should be downloaded even if it already exists in the local cache.

```bash
(main) $> gandlf hf --download --repo-id distilbert-base-uncased revision --revision v1.1 --token hf_**** --cache-dir ./path/to/cache --local-dir ./path/to/dir --force-download
```

### Upload an entire folder
Use the `gandlf hf --upload` upload command to upload files to the Hub directly.

```bash
(main) $> gandlf hf --upload --repo-id Wauplin/my-cool-model --folder-path ./model --token hf_****
```

### Upload to a Specific Path in Repo
Relative path of the directory in the repo. Will default to the root folder of the repository.
```bash
(main) $> gandlf hf --upload --repo-id Wauplin/my-cool-model --folder-path ./model/data --path-in-repo ./data --token hf_****
```

### Upload multiple files
To upload multiple files from a folder at once without uploading the entire folder, use the --allow-patterns and --ignore-patterns patterns. It can also be combined with the --delete-patterns option to delete files on the repo while uploading new ones. In the example below, we sync the local Space by deleting remote files and uploading all files except the ones in /logs:

```bash
(main) $> gandlf hf Wauplin/space-example --repo-type=space --exclude="/logs/*" --delete="*" --commit-message="Sync local Space with Hub"
```

### Specify a token
To upload files, you must use a token. By default, the token saved locally will be used. If you want to authenticate explicitly, use the --token option:

```bash
(main) $>gandlf hf --upload Wauplin/my-cool-model --folder-path ./model  --token=hf_****
```

### Specify a commit message
Use the --commit-message and --commit-description to set a custom message and description for your commit instead of the default one

```bash
(main) $>gandlf hf --upload Wauplin/my-cool-model --folder-path ./model  --token=hf_****
--commit-message "Epoch 34/50" --commit-description="Val accuracy: 68%. Check tensorboard for more details."
```

### Upload to a dataset or Space
To upload to a dataset or a Space, use the --repo-type option:

```bash
(main) $>gandlf hf --upload Wauplin/my-cool-model --folder-path ./model  --token=hf_****
--repo-type dataset
```

### Huggingface Template For Upload
#### Design and Modify Template 
To design the huggingface template use the hugging_face.md file change the mandatory field 
[REQUIRED_FOR_GANDLF] to it's respective name don't  leave it blank other wise it may through error, other field can be modeified by the user as per his convenience

```bash
# Here the required field change  from [REQUIRED_FOR_GANDLF] to [GANDLF]
**Developed by:** {{ developers | default("[GANDLF]", true)}}
```
#### Mentioned The Huggingface Template 
To mentioned the Huggingface Template , use the --hf-template option:

```bash
(main) $>gandlf hf --upload Wauplin/my-cool-model --folder-path ./model  --token=hf_****
--repo-type dataset --hf-template /hugging_face.md

```
