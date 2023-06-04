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

A major reason why one would want to anonymize data is to ensure that trained models do not inadvertently do not encode protect health information [[1](https://doi.org/10.1145/3436755),[2](https://doi.org/10.1038/s42256-020-0186-1)]. GaNDLF can anonymize single images or a collection of images using the `gandlf_anonymizer` script. It can be used as follows:

```bash
# continue from previous shell
(venv_gandlf) $> python gandlf_anonymizer
  # -h, --help         show help message and exit
  -c ./samples/config_anonymizer.yaml \ # anonymizer configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./input_dir_or_file \ # input directory containing series of images to anonymize or a single image
  -o ./output_dir_or_file # output directory to save anonymized images or a single output image file
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
(venv_gandlf) $> python gandlf_patchMiner \ 
  # -h, --help         show help message and exit
  -c ./exp_patchMiner/config.yaml \ # patch extraction configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./exp_patchMiner/input.csv \ # data in CSV format 
  -o ./exp_patchMiner/output_dir/ # output directory
```

### Running preprocessing before training/inference (optional)

Running preprocessing before training/inference is optional, but recommended. It will significantly reduce the computational footprint during training/inference at the expense of larger storage requirements. To run preprocessing before training/inference you can use the following command, which will save the processed data in `./experiment_0/output_dir/` with a new data CSV and the corresponding model configuration:

```bash
# continue from previous shell
(venv_gandlf) $> python gandlf_preprocess \
  # -h, --help         show help message and exit
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

- `Channel` can be substituted with `Modality` or `Image`
- `Label` can be substituted with `Mask` or `Segmentation`and is used to specify the annotation file for segmentation models
- For classification/regression, add a column called `ValueToPredict`. Currently, we are supporting only a single value prediction per model.
- Only a single `Label` or `ValueToPredict` header should be passed 
    - Multiple segmentation classes should be in a single file with unique label numbers.
    - Multi-label classification/regression is currently not supported.

### Using the `gandlf_constructCSV` application

To make the process of creating the CSV easier, we have provided a utility application called `gandlf_constructCSV`. This script works when the data is arranged in the following format (example shown of the data directory arrangement from the [Brain Tumor Segmentation (BraTS) Challenge](https://www.synapse.org/brats)):

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
(venv_gandlf) $> python gandlf_constructCSV \
  # -h, --help         show help message and exit
  -i $DATA_DIRECTORY # this is the main data directory 
  -c _t1.nii.gz,_t1ce.nii.gz,_t2.nii.gz,_flair.nii.gz \ # an example image identifier for 4 structural brain MR sequences for BraTS, and can be changed based on your data
  -l _seg.nii.gz \ # an example label identifier - not needed for regression/classification, and can be changed based on your data
  -o ./experiment_0/train_data.csv # output CSV to be used for training
```

**Notes**:
- For classification/regression, add a column called `ValueToPredict`. Currently, we are supporting only a single value prediction per model.
- `SubjectID` or `PatientName` is used to ensure that the randomized split is done per-subject rather than per-image.
- For data arrangement different to what is described above, a customized script will need to be written to generate the CSV, or you can enter the data manually into the CSV. 


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

1. The `gandlf_configGenerator` script can be used to generate a grid of configurations for tuning the hyperparameters of a baseline configuration that works for your dataset and problem. 
2. Use a strategy file (example is shown in [samples/config_generator_strategy.yaml](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_generator_sample_strategy.yaml).
3. Provide the baseline configuration which has enabled you to successfully train a model for `1` epoch for your dataset and problem at hand (regardless of the efficacy).
4. Run the following command:

```bash
# continue from previous shell
(venv_gandlf) $> python gandlf_configGenerator \
  # -h, --help         show help message and exit
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
(venv_gandlf) $> python gandlf_run \
  ## -h, --help         show help message and exit
  ## -v, --version      Show program's version number and exit.
  -c ./experiment_0/model.yaml \ # model configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./experiment_0/train.csv \ # data in CSV format 
  -m ./experiment_0/model_dir/ \ # model directory (i.e., the `modeldir`) where the output of the training will be stored, created if not present
  -t True \ # True == train, False == inference
  -d cuda # ensure CUDA_VISIBLE_DEVICES env variable is set for GPU device, use 'cpu' for CPU workloads
  # -rt , --reset # [optional] completely resets the previous run by deleting `modeldir`
  # -rm , --resume # [optional] resume previous training by only keeping model dict in `modeldir`
```

### Special notes for Inference for Histology images

- If you trying to perform inference on pre-extracted patches, please change the `modality` key in the configuration to `rad`. This will ensure the histology-specific pipelines are not triggered.
- However, if you are trying to perform inference on full WSIs, `modality` should be kept as `histo`.


## Generate Metrics 

GaNDLF provides a script to generate metrics after an inference process is done.The script can be used as follows:

```bash
# continue from previous shell
(venv_gandlf) $> python gandlf_generateMetrics \
  ## -h, --help         show help message and exit
  ## -v, --version      Show program's version number and exit.
  -c , --config       The configuration file (contains all the information related to the training/inference session)
  -i , --inputdata    CSV file that is used to generate the metrics; should contain 3 columns: 'subjectid, prediction, target'
  -o , --outputfile   Location to save the output dictionary. If not provided, will print to stdout.
```

Once you have your CSV in the specific format, you can pass it on to generate the metrics. Here is an example for segmentation:

```csv
SubjectID,Target,Prediction
001,/path/to/001/target.nii.gz,/path/to/001/prediction.nii.gz
002,/path/to/002/target.nii.gz,/path/to/001/prediction.nii.gz
...
```

Similarly for classification or regression (`A`, `B`, `C`, `D` are integers for classification and floats for regression):

```csv
SubjectID,Target,Prediction
001,A,B
002,C,D
...
```


## Parallelize the Training

### Multi-GPU training

GaNDLF enables relatively straightforward multi-GPU training. Simply set the `CUDA_VISIBLE_DEVICES` environment variable to the list of GPUs you want to use, and pass `cuda` as the device to the `gandlf_run` script. For example, if you want to use GPUs 0, 1, and 2, you would set `CUDA_VISIBLE_DEVICES=0,1,2` [[ref](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/)] and pass `-d cuda` to the `gandlf_run` script.

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
- The predictions will be saved in the same directory as the model if `outputdir` is not passed to `gandlf_run`.
- For segmentation, a directory will be created per subject ID in the input CSV.
- For classification/regression, the predictions will be generated in the `outputdir` or `modeldir` as a CSV file.


## Plot the final results

After the testing/validation training is finished, GaNDLF enables the collection of all the statistics from the final models for testing and validation datasets and plot them. The [gandlf_collectStats](https://github.com/mlcommons/GaNDLF/blob/master/gandlf_collectStats) can be used for plotting:

```bash
# continue from previous shell
(venv_gandlf) $> python gandlf_collectStats \
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

If you have a model previously trained using GaNDLF that you wish to run graph optimizations on, you can use the `gandlf_optimize` script to do so. The following command shows how it works:

```bash
# continue from previous shell
(venv_gandlf) $> python gandlf_optimizeModel \
  -m /path/to/trained/${architecture_name}_best.pth.tar \  # directory which contains testing and validation models
  -c ./experiment_0/config_used_to_train.yaml  # the config file used to train the model
```

If `${architecture_name}` is supported, the optimized model will get generated in the model directory, with the name `${architecture_name}_optimized.onnx`.

## Deployment

GaNDLF provides the ability to deploy models into easy-to-share, easy-to-use formats -- users of your model do not even need to install GaNDLF. Currently, Docker images are supported (which can be converted to [Apptainer/Singularity format](https://apptainer.org/docs/user/main/docker_and_oci.html)). These images meet [the MLCube interface](https://mlcommons.org/en/mlcube/). This allows your algorithm to be used in a consistent manner with other machine learning tools.

The resulting image contains your specific version of GaNDLF (including any custom changes you have made) and your trained model and configuration. This ensures that upstream changes to GaNDLF will not break compatibility with your model.

Please note that in order to deploy a model, for technical reasons, you need write access to the GaNDLF package. With a virtual environment this should be automatic. See the [installation instructions](./setup.md#installation).

To deploy a model, simply run the `gandlf_deploy` command after training a model. You will need the [Docker engine](https://www.docker.com/get-started/) installed to build Docker images. This will create the image and, for MLCubes, generate an MLCube directory complete with an `mlcube.yaml` specifications file, along with the workspace directory copied from a pre-existing template. 

```bash
# continue from previous shell
(venv_gandlf) $> python gandlf_deploy \
  ## -h, --help         show help message and exit
  -c ./experiment_0/model.yaml \ # Configuration to bundle with the model (you can recover it with gandlf_recoverConfig first if needed)
  -m ./experiment_0/model_dir/ \ # model directory (i.e., modeldir)
  --target docker \ # the target platform (--help will show all available targets)
  --mlcube-root ./my_new_mlcube_dir \ # Directory containing mlcube.yaml (used to configure your image base)
  -o ./output_dir # Output directory where a  new mlcube.yaml file to be distributed with your image will be created
```




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

Remember that the process running in the container only considers the filesystem inside the container, which is structured differently from that of your host machine. Therefore, you will need to give paths relative to the mount point *destination*. Additionally, any paths used internally by GaNDLF will refer to locations inside the container. This means that data CSVs produced by the `gandlf_constructCSV` script will need to be made from the container and with input in the same locations. Expanding on our last example:

```bash
(main) $> docker run -it --rm --name dataprep \
  --volume /home/researcher/gandlf_input:/input:ro \ # input data is mounted as read-only
  --volume /home/researcher/gandlf_output:/output \ # output data is mounted as read-write
  cbica/gandlf:latest-cpu \ # change to appropriate docker image tag
  gandlf_constructCSV \ # standard construct CSV API starts
  --inputDir /input/data \
  --outputFile /output/data.csv \
  --channelsID _t1.nii.gz \
  --labelID _seg.nii.gz
```

The previous command will generate a data CSV file that you can safely edit outside the container (such as by adding a `ValueToPredict` column). Then, you can refer to the same file when running again:

```bash
(main) $> docker run -it --rm --name training \
  --volume /home/researcher/gandlf_input:/input:ro \ # input data is mounted as read-only
  --volume /home/researcher/gandlf_output:/output \ # output data is mounted as read-write
  cbica/gandlf:latest-cpu \ # change to appropriate docker image tag
  gandlf_run --train True \ # standard training API starts
  --config /input/config.yml \
  --inputdata /output/data.csv \
  --modeldir /output/model
```
#### Special Case for Training

Considering that you want to train on an existing model that is inside the GaNDLF container (such as in an MLCube container created by `gandlf_deploy`), the output will be to a location embedded inside the container. Since you cannot mount something into that spot without overwriting the model, you can instead use the built-in `docker cp` command to extract the model afterward. For example, you can fine-tune a model on your own data using the following commands as a starting point:

```bash
# Run training on your new data
(main) $> docker run --name gandlf_training mlcommons/gandlf-pretrained:0.0.1 -v /my/input/data:/input gandlf_run -m /embedded_model/ [...] # Do not include "--rm" option!
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
(main) $> docker run --gpus all -e CUDA_VISIBLE_DEVICES -it --rm --name gandlf cbica/gandlf:latest-cuda113 gandlf_run --device cuda [...]
```

This can be replicated for ROCm for AMD , by following the [instructions to set up the ROCm Container Toolkit](https://rocmdocs.amd.com/en/latest/ROCm_Virtualization_Containers/ROCm-Virtualization-&-Containers.html?highlight=docker).

## MLCubes

GaNDLF, and GaNDLF-created models, may be distributed as an [MLCube](https://mlcommons.github.io/mlcube/). This involves distributing an `mlcube.yaml` file. That file can be specified when using the [MLCube runners](https://mlcommons.github.io/mlcube/runners/). The runner will perform many aspects of configuring your container for you. Currently, only the `mlcube_docker` runner is supported. 

See the [MLCube documentation](https://mlcommons.github.io/mlcube/) for more details.
