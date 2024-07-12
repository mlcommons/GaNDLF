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
  -d cuda # ensure CUDA_VISIBLE_DEVICES env variable is set for GPU device, use 'cpu' for CPU workloads
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

To generate image to image metrics for synthesis tasks (including for the BraTS synthesis tasks [[1](https://www.synapse.org/#!Synapse:syn51156910/wiki/622356), [2](https://www.synapse.org/#!Synapse:syn51156910/wiki/622357)]), ensure that the config has `problem_type: synthesis`, and the CSV can be in the same format as segmentation (note that the `Mask` column is optional):

```csv
SubjectID,Target,Prediction,Mask
001,/path/to/001/target_image.nii.gz,/path/to/001/prediction_image.nii.gz,/path/to/001/brain_mask.nii.gz
002,/path/to/002/target_image.nii.gz,/path/to/002/prediction_image.nii.gz,/path/to/002/brain_mask.nii.gz
...
```


## Parallelize the Training

### Multi-GPU training

GaNDLF enables relatively straightforward multi-GPU training. Simply set the `CUDA_VISIBLE_DEVICES` environment variable to the list of GPUs you want to use, and pass `cuda` as the device to the `gandlf run` command. For example, if you want to use GPUs 0, 1, and 2, you would set `CUDA_VISIBLE_DEVICES=0,1,2` [[ref](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/)] and pass `-d cuda` to the `gandlf run` command.

### Distributed training

Distributed training is a more difficult problem to address, since there are multiple ways to configure a high-performance computing cluster (SLURM, OpenHPC, Kubernetes, and so on). Owing to this discrepancy, we have ensured that GaNDLF allows multiple training jobs to be submitted in relatively straightforward manner using the command line inference of each site’s configuration. Simply populate the `parallel_compute_command` in the [configuration](#customize-the-training) with the specific command to run before the training job, and GaNDLF will use this string to submit the training job. 


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

### Deploy as a Model

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

Once you have a model definition, it is easy to perform federated learning using the [Open Federated Learning (OpenFL) library](https://github.com/securefederatedai/openfl). Follow the tutorial in [this page](https://openfl.readthedocs.io/en/latest/running_the_federation_with_gandlf.html) to get started.


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

## Hugging Face Hub CLI

This tool allows you to interact with the Hugging Face Hub directly from a terminal. For example, you can login to your account, create a repository, upload and download files, etc. It also comes with handy features to configure your machine or manage your cache.

### huggingface-cli login

In many cases, you must be logged in to a Hugging Face account to interact with the Hub (download private repos, upload files, create PRs, etc.). To do so, you need a [User Access Token](https://huggingface.co/docs/hub/security-tokens) from your [Settings page](https://huggingface.co/settings/tokens). The User Access Token is used to authenticate your identity to the Hub. Make sure to set a token with write access if you want to upload or modify content.

Once you have your token, run the following command in your terminal:

```bash
>>> gandlf hf login
```

This command will prompt you for a token. Copy-paste yours and press *Enter*. Then you'll be asked if the token should also be saved as a git credential. Press *Enter* again (default to yes) if you plan to use `git` locally. Finally, it will call the Hub to check that your token is valid and save it locally.

```
_|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
_|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
_|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
_|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
_|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Token:
Add token as git credential? (Y/n)
Token is valid (permission: write).
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/wauplin/.cache/huggingface/token
Login successful
```

Alternatively, if you want to log-in without being prompted, you can pass the token directly from the command line. To be more secure, we recommend passing your token as an environment variable to avoid pasting it in your command history.

```bash
# Or using an environment variable
>>> gandlf hf login --token $HUGGINGFACE_TOKEN --add-to-git-credential
Token is valid (permission: write).
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/wauplin/.cache/huggingface/token
Login successful
```

For more details about authentication, check out [this section](../quick-start#authentication).

### huggingface-cli whoami

If you want to know if you are logged in, you can use `gandlf hf whoami`. This command doesn't have any options and simply prints your username and the organizations you are a part of on the Hub:

```bash
gandlf hf whoami
Wauplin
orgs:  huggingface,eu-test,OAuthTesters,hf-accelerate,HFSmolCluster
```

If you are not logged in, an error message will be printed.

### huggingface-cli logout

This commands logs you out. In practice, it will delete the token saved on your machine.

This command will not log you out if you are logged in using the `HF_TOKEN` environment variable (see [reference](../package_reference/environment_variables#hftoken)). If that is the case, you must unset the environment variable in your machine configuration.

### huggingface-cli download


Use the `gandlf hf download` command to download files from the Hub directly. Internally, it uses the same [`hf_hub_download`] and [`snapshot_download`] helpers described in the [Download](./download) guide and prints the returned path to the terminal. In the examples below, we will walk through the most common use cases. For a full list of available options, you can run:

```bash
gandlf hf download --help
```

### Download a single file

To download a single file from a repo, simply provide the repo_id and filename as follow:

```bash
>>> gandlf hf download gpt2 config.json
downloading https://huggingface.co/gpt2/resolve/main/config.json to /home/wauplin/.cache/huggingface/hub/tmpwrq8dm5o
(…)ingface.co/gpt2/resolve/main/config.json: 100%|██████████████████████████████████| 665/665 [00:00<00:00, 2.49MB/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

The command will always print on the last line the path to the file on your local machine.

### Download an entire repository

In some cases, you just want to download all the files from a repository. This can be done by just specifying the repo id:

```bash
>>> gandlf hf download HuggingFaceH4/zephyr-7b-beta
Fetching 23 files:   0%|                                                | 0/23 [00:00<?, ?it/s]
...
...
/home/wauplin/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/3bac358730f8806e5c3dc7c7e19eb36e045bf720
```

### Download multiple files

You can also download a subset of the files from a repository with a single command. This can be done in two ways. If you already have a precise list of the files you want to download, you can simply provide them sequentially:

```bash
>>> gandlf hf download gpt2 config.json model.safetensors
Fetching 2 files:   0%|                                                                        | 0/2 [00:00<?, ?it/s]
downloading https://huggingface.co/gpt2/resolve/11c5a3d5811f50298f278a704980280950aedb10/model.safetensors to /home/wauplin/.cache/huggingface/hub/tmpdachpl3o
(…)8f278a7049802950aedb10/model.safetensors: 100%|██████████████████████████████| 8.09k/8.09k [00:00<00:00, 40.5MB/s]
Fetching 2 files: 100%|████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.76it/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

The other approach is to provide patterns to filter which files you want to download using `--include` and `--exclude`. For example, if you want to download all safetensors files from [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), except the files in FP16 precision:

```bash
>>> gandlf hf download stabilityai/stable-diffusion-xl-base-1.0 --include "*.safetensors" --exclude "*.fp16.*"*
Fetching 8 files:   0%|                                                                         | 0/8 [00:00<?, ?it/s]
...
...
Fetching 8 files: 100%|█████████████████████████████████████████████████████████████████████████| 8/8 (...)
/home/wauplin/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b
```

### Download a dataset or a Space

The examples above show how to download from a model repository. To download a dataset or a Space, use the `--repo-type` option:

```bash
# https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
>>> gandlf hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

# https://huggingface.co/spaces/HuggingFaceH4/zephyr-chat
>>> gandlf hf download HuggingFaceH4/zephyr-chat --repo-type space

...
```

### Download a specific revision

The examples above show how to download from the latest commit on the main branch. To download from a specific revision (commit hash, branch name or tag), use the `--revision` option:

```bash
>>> gandlf hf download bigcode/the-stack --repo-type dataset --revision v1.1
...
```

### Download to a local folder

The recommended (and default) way to download files from the Hub is to use the cache-system. However, in some cases you want to download files and move them to a specific folder. This is useful to get a workflow closer to what git commands offer. You can do that using the `--local-dir` option.

A `./huggingface/` folder is created at the root of your local directory containing metadata about the downloaded files. This prevents re-downloading files if they're already up-to-date. If the metadata has changed, then the new file version is downloaded. This makes the `local-dir` optimized for pulling only the latest changes.

<Tip>

For more details on how downloading to a local file works, check out the [download](./download.md#download-files-to-a-local-folder) guide.

</Tip>

```bash
>>> gandlf hf download adept/fuyu-8b model-00001-of-00002.safetensors --local-dir fuyu
...
fuyu/model-00001-of-00002.safetensors
```

### Specify cache directory

If not using `--local-dir`, all files will be downloaded by default to the cache directory defined by the `HF_HOME` [environment variable](../package_reference/environment_variables#hfhome). You can specify a custom cache using `--cache-dir`:

```bash
>>> gandlf hf download adept/fuyu-8b --cache-dir ./path/to/cache
...
./path/to/cache/models--adept--fuyu-8b/snapshots/ddcacbcf5fdf9cc59ff01f6be6d6662624d9c745
```

### Specify a token

To access private or gated repositories, you must use a token. By default, the token saved locally (using `gandlf hf login`) will be used. If you want to authenticate explicitly, use the `--token` option:

```bash
>>> gandlf hf download gpt2 config.json --token=hf_****
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

### Quiet mode

By default, the `gandlf hf download` command will be verbose. It will print details such as warning messages, information about the downloaded files, and progress bars. If you want to silence all of this, use the `--quiet` option. Only the last line (i.e. the path to the downloaded files) is printed. This can prove useful if you want to pass the output to another command in a script.

```bash
>>> gandlf hf download gpt2 --quiet
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

### Download timeout

On machines with slow connections, you might encounter timeout issues like this one:
```bash
`requests.exceptions.ReadTimeout: (ReadTimeoutError("HTTPSConnectionPool(host='cdn-lfs-us-1.huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: a33d910c-84c6-4514-8362-c705e2039d38)')`
```

To mitigate this issue, you can set the `HF_HUB_DOWNLOAD_TIMEOUT` environment variable to a higher value (default is 10):
```bash
export HF_HUB_DOWNLOAD_TIMEOUT=30
```

For more details, check out the [environment variables reference](../package_reference/environment_variables#hfhubdownloadtimeout).And rerun your download command.

## huggingface-cli upload

Use the `gandlf hf upload` command to upload files to the Hub directly. Internally, it uses the same [`upload_file`] and [`upload_folder`] helpers described in the [Upload](./upload) guide. In the examples below, we will walk through the most common use cases. For a full list of available options, you can run:

```bash
>>> gandlf hf upload --help
```

### Upload an entire folder

The default usage for this command is:

```bash
# Usage:  gandlf hf upload [repo_id] [local_path] [path_in_repo]
```

To upload the current directory at the root of the repo, use:

```bash
>>> gandlf hf upload my-cool-model . .
https://huggingface.co/Wauplin/my-cool-model/tree/main/
```

<Tip>

If the repo doesn't exist yet, it will be created automatically.

</Tip>

You can also upload a specific folder:

```bash
>>> gandlf hf upload my-cool-model ./models .
https://huggingface.co/Wauplin/my-cool-model/tree/main/
```

Finally, you can upload a folder to a specific destination on the repo:

```bash
>>> gandlf hf upload my-cool-model ./path/to/curated/data /data/train
https://huggingface.co/Wauplin/my-cool-model/tree/main/data/train
```

### Upload a single file

You can also upload a single file by setting `local_path` to point to a file on your machine. If that's the case, `path_in_repo` is optional and will default to the name of your local file:

```bash
>>> gandlf hf upload Wauplin/my-cool-model ./models/model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/model.safetensors
```

If you want to upload a single file to a specific directory, set `path_in_repo` accordingly:

```bash
>>> gandlf hf upload Wauplin/my-cool-model ./models/model.safetensors /vae/model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/vae/model.safetensors
```

### Upload multiple files

To upload multiple files from a folder at once without uploading the entire folder, use the `--include` and `--exclude` patterns. It can also be combined with the `--delete` option to delete files on the repo while uploading new ones. In the example below, we sync the local Space by deleting remote files and uploading all files except the ones in `/logs`:

```bash
# Sync local Space with Hub (upload new files except from logs/, delete removed files)
>>> gandlf hf upload Wauplin/space-example --repo-type=space --exclude="/logs/*" --delete="*" --commit-message="Sync local Space with Hub"
...
```

### Upload to a dataset or Space

To upload to a dataset or a Space, use the `--repo-type` option:

```bash
>>> gandlf hf upload Wauplin/my-cool-dataset ./data /train --repo-type=dataset
...
```

### Upload to an organization

To upload content to a repo owned by an organization instead of a personal repo, you must explicitly specify it in the `repo_id`:

```bash
>>> gandlf hf upload MyCoolOrganization/my-cool-model . .
https://huggingface.co/MyCoolOrganization/my-cool-model/tree/main/
```

### Upload to a specific revision

By default, files are uploaded to the `main` branch. If you want to upload files to another branch or reference, use the `--revision` option:

```bash
# Upload files to a PR
>>> gandlf hf upload bigcode/the-stack . . --repo-type dataset --revision refs/pr/104
...
```

**Note:** if `revision` does not exist and `--create-pr` is not set, a branch will be created automatically from the `main` branch.

### Upload and create a PR

If you don't have the permission to push to a repo, you must open a PR and let the authors know about the changes you want to make. This can be done by setting the `--create-pr` option:

```bash
# Create a PR and upload the files to it
>>> gandlf hf upload bigcode/the-stack . . --repo-type dataset --revision refs/pr/104
https://huggingface.co/datasets/bigcode/the-stack/blob/refs%2Fpr%2F104/
```

### Upload at regular intervals

In some cases, you might want to push regular updates to a repo. For example, this is useful if you're training a model and you want to upload the logs folder every 10 minutes. You can do this using the `--every` option:

```bash
# Upload new logs every 10 minutes
gandlf hf upload training-model logs/ --every=10
```

### Specify a commit message

Use the `--commit-message` and `--commit-description` to set a custom message and description for your commit instead of the default one

```bash
>>> gandlf hf upload Wauplin/my-cool-model ./models . --commit-message="Epoch 34/50" --commit-description="Val accuracy: 68%. Check tensorboard for more details."
...
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

### Specify a token

To upload files, you must use a token. By default, the token saved locally (using `gandlf hf login`) will be used. If you want to authenticate explicitly, use the `--token` option:

```bash
>>> gandlf hf upload Wauplin/my-cool-model ./models . --token=hf_****
...
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

### Quiet mode

By default, the `gandlf hf upload` command will be verbose. It will print details such as warning messages, information about the uploaded files, and progress bars. If you want to silence all of this, use the `--quiet` option. Only the last line (i.e. the URL to the uploaded files) is printed. This can prove useful if you want to pass the output to another command in a script.

```bash
>>> gandlf hf upload Wauplin/my-cool-model ./models . --quiet
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

## huggingface-cli repo-files

If you want to delete files from a Hugging Face repository, use the `gandlf hf repo-files` command. 

### Delete files

The `gandlf hf repo-files <repo_id> delete` sub-command allows you to delete files from a repository. Here are some usage examples.

Delete a folder :
```bash
>>> gandlf hf repo-files Wauplin/my-cool-model delete folder/  
Files correctly deleted from repo. Commit: https://huggingface.co/Wauplin/my-cool-mo...
```

Delete multiple files: 
```bash
>>> gandlf hf repo-files Wauplin/my-cool-model delete file.txt folder/pytorch_model.bin
Files correctly deleted from repo. Commit: https://huggingface.co/Wauplin/my-cool-mo...
```

Use Unix-style wildcards to delete sets of files: 
```bash
>>> gandlf hf repo-files Wauplin/my-cool-model delete *.txt folder/*.bin 
Files correctly deleted from repo. Commit: https://huggingface.co/Wauplin/my-cool-mo...
```

### Specify a token

To delete files from a repo you must be authenticated and authorized. By default, the token saved locally (using `gandlf hf login`) will be used. If you want to authenticate explicitly, use the `--token` option:

```bash
>>> gandlf hf repo-files --token=hf_**** Wauplin/my-cool-model delete file.txt 
```

## huggingface-cli scan-cache

Scanning your cache directory is useful if you want to know which repos you have downloaded and how much space it takes on your disk. You can do that by running `gandlf hf scan-cache`:

```bash
>>> gandlf hf scan-cache
REPO ID                     REPO TYPE SIZE ON DISK NB FILES LAST_ACCESSED LAST_MODIFIED REFS                LOCAL PATH
--------------------------- --------- ------------ -------- ------------- ------------- ------------------- -------------------------------------------------------------------------
glue                        dataset         116.3K       15 4 days ago    4 days ago    2.4.0, main, 1.17.0 /home/wauplin/.cache/huggingface/hub/datasets--glue
google/fleurs               dataset          64.9M        6 1 week ago    1 week ago    refs/pr/1, main     /home/wauplin/.cache/huggingface/hub/datasets--google--fleurs
Jean-Baptiste/camembert-ner model           441.0M        7 2 weeks ago   16 hours ago  main                /home/wauplin/.cache/huggingface/hub/models--Jean-Baptiste--camembert-ner
bert-base-cased             model             1.9G       13 1 week ago    2 years ago                       /home/wauplin/.cache/huggingface/hub/models--bert-base-cased
t5-base                     model            10.1K        3 3 months ago  3 months ago  main                /home/wauplin/.cache/huggingface/hub/models--t5-base
t5-small                    model           970.7M       11 3 days ago    3 days ago    refs/pr/1, main     /home/wauplin/.cache/huggingface/hub/models--t5-small

Done in 0.0s. Scanned 6 repo(s) for a total of 3.4G.
Got 1 warning(s) while scanning. Use -vvv to print details.
```

For more details about how to scan your cache directory, please refer to the [Manage your cache](./manage-cache#scan-cache-from-the-terminal) guide.

## huggingface-cli delete-cache

`gandlf hf delete-cache` is a tool that helps you delete parts of your cache that you don't use anymore. This is useful for saving and freeing disk space. To learn more about using this command, please refer to the [Manage your cache](./manage-cache#clean-cache-from-the-terminal) guide.

## huggingface-cli tag

The `gandlf hf tag` command allows you to tag, untag, and list tags for repositories.

### Tag a model

To tag a repo, you need to provide the `repo_id` and the `tag` name:

```bash
>>> gandlf hf tag Wauplin/my-cool-model v1.0
You are about to create tag v1.0 on model Wauplin/my-cool-model
Tag v1.0 created on Wauplin/my-cool-model
```

### Tag a model at a specific revision

If you want to tag a specific revision, you can use the `--revision` option. By default, the tag will be created on the `main` branch:

```bash
>>> gandlf hf tag Wauplin/my-cool-model v1.0 --revision refs/pr/104
You are about to create tag v1.0 on model Wauplin/my-cool-model
Tag v1.0 created on Wauplin/my-cool-model
```

### Tag a dataset or a Space

If you want to tag a dataset or Space, you must specify the `--repo-type` option:

```bash
>>> gandlf hf tag bigcode/the-stack v1.0 --repo-type dataset
You are about to create tag v1.0 on dataset bigcode/the-stack
Tag v1.0 created on bigcode/the-stack
```

### List tags

To list all tags for a repository, use the `-l` or `--list` option:

```bash
>>> gandlf hf tag Wauplin/gradio-space-ci -l --repo-type space
Tags for space Wauplin/gradio-space-ci:
0.2.2
0.2.1
0.2.0
0.1.2
0.0.2
0.0.1
```

### Delete a tag

To delete a tag, use the `-d` or `--delete` option:

```bash
>>> gandlf hf tag -d Wauplin/my-cool-model v1.0
You are about to delete tag v1.0 on model Wauplin/my-cool-model
Proceed? [Y/n] y
Tag v1.0 deleted on Wauplin/my-cool-model
```

You can also pass `-y` to skip the confirmation step.

## huggingface-cli env

The `gandlf hf env` command prints details about your machine setup. This is useful when you open an issue on [GitHub](https://github.com/huggingface/huggingface_hub) to help the maintainers investigate your problem.

```bash
>>> gandlf hf env

Copy-and-paste the text below in your GitHub issue.

- huggingface_hub version: 0.19.0.dev0
- Platform: Linux-6.2.0-36-generic-x86_64-with-glibc2.35
- Python version: 3.10.12
- Running in iPython ?: No
- Running in notebook ?: No
- Running in Google Colab ?: No
- Token path ?: /home/wauplin/.cache/huggingface/token
- Has saved token ?: True
- Who am I ?: Wauplin
- Configured git credential helpers: store
- FastAI: N/A
- Tensorflow: 2.11.0
- Torch: 1.12.1
- Jinja2: 3.1.2
- Graphviz: 0.20.1
- Pydot: 1.4.2
- Pillow: 9.2.0
- hf_transfer: 0.1.3
- gradio: 4.0.2
- tensorboard: 2.6
- numpy: 1.23.2
- pydantic: 2.4.2
- aiohttp: 3.8.4
- ENDPOINT: https://huggingface.co
- HF_HUB_CACHE: /home/wauplin/.cache/huggingface/hub
- HF_ASSETS_CACHE: /home/wauplin/.cache/huggingface/assets
- HF_TOKEN_PATH: /home/wauplin/.cache/huggingface/token
- HF_HUB_OFFLINE: False
- HF_HUB_DISABLE_TELEMETRY: False
- HF_HUB_DISABLE_PROGRESS_BARS: None
- HF_HUB_DISABLE_SYMLINKS_WARNING: False
- HF_HUB_DISABLE_EXPERIMENTAL_WARNING: False
- HF_HUB_DISABLE_IMPLICIT_TOKEN: False
- HF_HUB_ENABLE_HF_TRANSFER: False
- HF_HUB_ETAG_TIMEOUT: 10
- HF_HUB_DOWNLOAD_TIMEOUT: 10
```