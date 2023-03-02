For any DL pipeline, the following flow needs to be performed:

1. Data preparation
2. Split data into training, validation, and testing
3. Customize the training parameters

A detailed data flow diagram is presented in [this link](https://github.com/mlcommons/GaNDLF/blob/master/docs/README.md#flowchart).

GaNDLF tackles all of these and the details are split in the manner explained in [the following section](#table-of-contents).
## Table of Contents

- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Preparing the Data](#preparing-the-data)
  - [Anonymize Data](#anonymize-data)
  - [Cleanup/Harmonize Data](#cleanupharmonize-data)
  - [Offline Patch Extraction (for histology images only)](#offline-patch-extraction-for-histology-images-only)
  - [Running preprocessing before training/inference](#running-preprocessing-before-traininginference)
- [Constructing the Data CSV](#constructing-the-data-csv)
- [Customize the Training](#customize-the-training)
  - [Running multiple experiments \[OPTIONAL\]](#running-multiple-experiments-optional)
- [Running GaNDLF (Training/Inference)](#running-gandlf-traininginference)
  - [Special note for Inference for Histology images](#special-note-for-inference-for-histology-images)
- [Parallelize the Training](#parallelize-the-training)
- [Expected Output(s)](#expected-outputs)
- [Plot the final results](#plot-the-final-results)
  - [Multi-GPU systems](#multi-gpu-systems)
- [M3D-CAM usage](#m3d-cam-usage)
- [Deployment](#deployment)
- [Running with Docker](#running-with-docker)
  - [Mounting Input and Output](#mounting-input-and-output)
    - [Special Case for Training](#special-case-for-training)
  - [Enabling GPUs](#enabling-gpus)
- [MLCubes](#mlcubes)


## Installation

Please follow the [installation instructions](./setup.md) to install GaNDLF. This should end up with the shell that looks like this with the GaNDLF virtual environment activated:

```bash
(venv_gandlf) $> ### subsequent commands go here
```

[Back To Top &uarr;](#table-of-contents)


## Preparing the Data

### Anonymize Data

GaNDLF can anonymize single images or a collection of images using the `gandlf_anonymizer` script. The usage is as follows:
```bash
(venv_gandlf) $> python gandlf_anonymizer
  # -h, --help         show help message and exit
  -c ./samples/config_anonymizer.yaml \ # anonymizer configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./input_dir_or_file \ # input directory containing series of images to anonymize or a single image
  -o ./output_dir_or_file # output directory to save anonymized images or a single output image file
```

[Back To Top &uarr;](#table-of-contents)

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

[Back To Top &uarr;](#table-of-contents)

### Offline Patch Extraction (for histology images only)

GaNDLF can be used to convert a Whole Slide Image (WSI) with or without a corresponding label map to patches using GaNDLF's integrated patch miner:

- Construct a YAML configuration for OPM with a minimum of the following keys:
  - `scale`: scale at which operations such as tissue mask calculation happens; defaults to 16
  - `patch_size`: defines the size of the patches to extract, should be a tuple type of integers (e.g., [256,256]) or a string containing patch size in microns (e.g., "[100m,100m]")
  - `num_patches`: defines the number of patches to extract; use -1 to mine until exhaustion
- A CSV file with the following columns:
  - `SubjectID`: the ID of the subject for the WSI
  - `Channel_0`: the WSI file
  - `Label`: (optional) the label map file
  - `value_map`: mapping RGB values in label image to integer values for training; defaults to None
  - `read_type`: either "random" or "sequential" (latter is more efficient); defaults to "random"
  - `pen_size_threshold`: thickness of pen strokes to be considered as a mask
  - `min_color_difference`: color difference between tissue and pen markings
  - `overlap_factor`: Portion of patches that are allowed to overlap (0->1); defaults to "0.0"
- Run the following command:
```bash
(venv_gandlf) $> python gandlf_patchMiner
  # -h, --help         show help message and exit
  -c ./exp_patchMiner/config.yaml \ # patch extraction configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./exp_patchMiner/input.csv \ # data in CSV format 
  -o ./exp_patchMiner/output_dir/ \ # output directory
```

[Back To Top &uarr;](#table-of-contents)

### Running preprocessing before training/inference

This is optional, but recommended. It will significantly reduce the computational footprint during training/inference at the expense of larger storage requirements.
```bash
# continue from previous shell
(venv_gandlf) $> python gandlf_preprocess \
  # -h, --help         show help message and exit
  -c ./experiment_0/model.yaml \ # model configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./experiment_0/train.csv \ # data in CSV format 
  -o ./experiment_0/output_dir/ \ # output directory
```

This will save the processed data in `./experiment_0/output_dir/` with a new data CSV and the corresponding model configuration.

[Back To Top &uarr;](#table-of-contents)


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

The [gandlf_constructCSV](https://github.com/mlcommons/GaNDLF/blob/master/gandlf_constructCSV) can be used to make this easier:

```bash
# continue from previous shell
(venv_gandlf) $> python gandlf_constructCSV \
  # -h, --help         show help message and exit
  -i ./experiment_0/data_dir/ # this is the main data directory
  -c _t1.nii.gz,_t1ce.nii.gz,_t2.nii.gz,_flair.nii.gz \ # an example image identifier for 4 structural brain MR sequences
  -l _seg.nii.gz # an example label identifier - not needed for regression/classification
  -o ./experiment_0/train_data.csv \ # output CSV to be used for training
```
**Note** that this cannot be used for classification/regression tasks directly, and will need modification based on the way your data is stored.

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

GaNDLF requires a YAML-based configuration that controls various aspects of the training/inference process. There are multiple samples for users to start as their baseline for further customization. The following is a list of the available samples:

- [Sample showing all the available options](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_all_options.yaml)
- [Segmentation example](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_segmentation_brats.yaml)
- [Regression example](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_regression.yaml)
- [Classification example](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_classification.yaml)

**Notes**: 
- More details on the configuration options are available in the [customization page](customize.md).
- Ensure that the configuration has valid syntax by checking the file using any YAML validator such as [yamlchecker.com](https://yamlchecker.com/) or [yamlvalidator.com](https://yamlvalidator.com/) **before** trying to train.

[Back To Top &uarr;](#table-of-contents)

### Running multiple experiments [OPTIONAL]

- The `gandlf_configGenerator` script can be used to generate a grid of configurations for tuning the hyperparameters of a baseline configuration that works for your dataset and problem. 
- Use a strategy file (example is shown in [samples/config_generator_strategy.yaml](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_generator_sample_strategy.yaml).
- Provide the baseline configuration which has enabled you to train a model for 1 epoch for your dataset and problem (regardless of the efficacy).
- Run the following command:  
```bash
(venv_gandlf) $> python gandlf_configGenerator \
  # -h, --help         show help message and exit
  -c ./samples/config_all_options.yaml \ # baseline configuration
  -s ./samples/config_generator_strategy.yaml \ # strategy file
  -o ./all_experiments/ # output directory
```
- For example, to generate 4 configurations that leverage `unet` and `resunet` architectures for learning rates of `[0.1,0.01]`,  you can pass the following strategy file:
```yaml
model:
  {
    architecture: [unet, resunet],
  }
learning_rate: [0.1, 0.01]
```

[Back To Top &uarr;](#table-of-contents)


## Running GaNDLF (Training/Inference)

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

### Special note for Inference for Histology images

- If you trying to perform inference on pre-extracted patches, please change the `modality` key in the configuration to `rad`, as this will ensure the histology-specific pipelines are not triggered.
- However, if you are trying to perform inference on full WSIs, `modality` should be kept as `histo`.

[Back To Top &uarr;](#table-of-contents)


## Parallelize the Training

GaNDLF allows multi-GPU training relatively easily. Simply set the `CUDA_VISIBLE_DEVICES` environment variable to the list of GPUs you want to use, and pass `cuda` as the device to the `gandlf_run` script. For example, if you want to use GPUs 0, 1, and 2, you would set `CUDA_VISIBLE_DEVICES=0,1,2` and pass `-d cuda` to the `gandlf_run` script.

Distributed training is a more difficult problem to address, since there are multiple ways to configure a high-performance computing cluster (SLURM, OpenHPC, Kubernetes, and so on). Owing to this discrepancy, we have ensured that GaNDLF allows multiple training jobs to be submitted in relatively straightforward manner using the command line inference of each site’s configuration. Simply populate the `paralle_compute_command` in the configuration with the specific command to run before the training job, and GaNDLF will use this string to submit the training job. 

[Back To Top &uarr;](#table-of-contents)


## Expected Output(s)

Once your model is trained, you should see the following output:

```bash
# continue from previous shell
(venv_gandlf) $> ls ./experiment_0/model_dir/
data_{cohort_type}.csv  # data CSV used for the different cohorts, which can be either training/validation/testing
data_{cohort_type}.pkl  # same as above, but in pickle format
logs_{cohort_type}.csv  # logs for the different cohorts that contain the various metrics, which can be either training/validation/testing
{architecture_name}_best.pth.tar # the best model in native PyTorch format
{architecture_name}_latest.pth.tar # the latest model in native PyTorch format
{architecture_name}_initial.pth.tar # the initial model in native PyTorch format
{architecture_name}_initial.{xml/bin} # the graph-optimized best model in ONNX format
# other files dependent on if training/validation/testing output was enabled in configuration
```

[Back To Top &uarr;](#table-of-contents)


## Plot the final results

After the testing/validation training is finished, GaNDLF makes it possible to collect all the statistics from the final models for testing and validation datasets and plot them. The [gandlf_collectStats](https://github.com/mlcommons/GaNDLF/blob/master/gandlf_collectStats) can be used for this:

```bash
# continue from previous shell
python gandlf_collectStats \
  -m /path/to/trained/models \  # directory which contains testing and validation models
  -o ./experiment_0/output_dir_stats/  # output directory to save stats and plot
```

[Back To Top &uarr;](#table-of-contents)

### Multi-GPU systems

Please ensure that the environment variable `CUDA_VISIBLE_DEVICES` is set [[ref](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/)].

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

[Back To Top &uarr;](#table-of-contents)


## Deployment

You can deploy models trained with GaNDLF into easy-to-share, easy-to-use formats -- users of your model do not even need to install GaNDLF.
Currently, Docker images are supported (which can be converted to Singularity format).
These images meet [the MLCube interface](https://mlcommons.org/en/mlcube/).
This allows your algorithm to be used in a consistent manner with other machine learning tools.

The resulting image contains your specific version of GaNDLF (including any custom changes you have made) and your trained model and configuration.
This ensures that upstream changes to GaNDLF will not break compatibility with your model.

To deploy a model, simply run the `gandlf_deploy` command after training a model. You will need the [Docker engine](https://www.docker.com/get-started/) installed to build Docker images.
This will create the image and, for MLCubes, generate an MLCube directory complete with an `mlcube.yaml` specifications file, along with the workspace directory copied from a pre-existing template. 

```bash
python gandlf_deploy \
  ## -h, --help         show help message and exit
  -c ./experiment_0/model.yaml \ # Configuration to bundle with the model (you can recover it with gandlf_recoverConfig first if needed)
  -m ./experiment_0/model_dir/ \ # model directory (i.e., modeldir)
  --target docker # the target platform (--help will show all available targets)
  --mlcube-root ./my_new_mlcube_dir \ # Directory containing mlcube.yaml (used to configure your image base)
  -o ./output_dir # Output directory where a  new mlcube.yaml file to be distributed with your image will be created
```

[Back To Top &uarr;](#table-of-contents)


## Running with Docker

Usage of GaNDLF remains generally the same even from Docker, but there are a few extra considerations.

Once you have pulled the GaNDLF image, it will have a tag, like "cbica/gandlf:latest-cpu". 
Run the following command to list your images and ensure GaNDLF is present:
```bash
docker image ls
```
You can invoke "docker run" with the appropriate tag to run GaNDLF:
```bash
docker run -it --rm --name gandlf cbica/gandlf:latest-cpu [gandlf command and parameters go here!]
```
Remember that arguments/options for *Docker itself* go *before* the image tag, while the command and arguments for GaNDLF go *after* the image tag.
For more details and options, see the [Docker run documentation](https://docs.docker.com/engine/reference/commandline/run/).

However, most commands that require files or directories as input or output will fail, because the container, by default, cannot read or write files on your machine for security reasons.
To fix this, we need to use mounts. 

[Back To Top &uarr;](#table-of-contents)

### Mounting Input and Output

The container is basically a filesystem of its own. To make your data available to the container, you will need to mount in files and folders.
Generally, it is useful to mount at least input folder (as readonly) and an output folder.
See the [Docker bind mount instructions](https://docs.docker.com/storage/bind-mounts/) for more information.

For example, you might run:
```bash
docker run -it --rm --name gandlf --volume /home/researcher/gandlf_input:/input:ro --volume /home/researcher/gandlf_output:/output cbica/gandlf:latest-cpu [command and args go here]
```

Remember that the process running in the container sees only the filesystem inside the container, which is structured differently from that of your host machine.
So you will need to give paths relative to the mount point *destination*.
Additionally, any paths used internally by GaNDLF will refer to locations inside the container.
This means that data CSVs produced by the gandlf_constructCSV script will need to be made from the container and with input in the same locations. Expanding on our last example:

```bash
docker run -it --rm --name dataprep \
  --volume /home/researcher/gandlf_input:/input:ro \ # input data is mounted as read-only
  --volume /home/researcher/gandlf_output:/output \ # output data is mounted as read-write
  cbica/gandlf:latest-cpu \ # change to appropriate docker image tag
  gandlf_constructCSV \ # standard construct CSV API starts
  --inputDir /input/data \
  --outputFile /output/data.csv \
  --channelsID _t1.nii.gz \
  --labelID _seg.nii.gz
```
The above command will generate a data CSV file that you can safely edit outside the container (such as by adding a ValueToPredict column).
Then, we can reference the same file when running again:

```bash
docker run -it --rm --name training \
  --volume /home/researcher/gandlf_input:/input:ro \ # input data is mounted as read-only
  --volume /home/researcher/gandlf_output:/output \ # output data is mounted as read-write
  cbica/gandlf:latest-cpu \ # change to appropriate docker image tag
  gandlf_run --train True \ # standard training API starts
  --config /input/config.yml \
  --inputdata /output/data.csv \
  --modeldir /output/model
```

[Back To Top &uarr;](#table-of-contents)
#### Special Case for Training

In the case where you want to train on an existing model that is inside the GaNDLF container (such as in an MLCube container created by `gandlf_deploy`), the output will be to a location embedded inside the container. Because you cannot mount something into that spot without overwriting the model, you can instead use the built-in `docker cp` command to extract the model afterward.

For example, you can fine-tune a model on your own data using the following commands as a starting point:
```bash
# Run training on your new data
docker run --name gandlf_training mlcommons/gandlf-pretrained:0.0.1 -v /my/input/data:/input gandlf_run -m /embedded_model/ [...] # Do not include "--rm" option!
# Copy the finetuned model out of the container, to a location on the host
docker cp gandlf_training:/embedded_model /home/researcher/extracted_model
# Now you can remove the container to clean up
docker rm -f gandlf_training
```

### Enabling GPUs

Some special arguments need to be passed to Docker to enable it to use your GPU.
With Docker version > 19.03 You can pass the "--gpus all" parameter to "docker run" to expose all GPUs to the container.
See the [NVIDIA Docker documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration) for more details.

If using CUDA, GaNDLF also expects the environment variable CUDA_VISIBLE_DEVICES to be set.
To use the same settings as your host machine, simply add "-e CUDA_VISIBLE_DEVICES" to your docker run command.

For example:
```bash
docker run --gpus all -e CUDA_VISIBLE_DEVICES -it --rm --name gandlf cbica/gandlf:latest-cuda113 gandlf_run --device cuda [...]
```

[Back To Top &uarr;](#table-of-contents)

## MLCubes

GaNDLF, and GaNDLF-created models, may be distributed as an [MLCube](https://mlcommons.github.io/mlcube/).
This involves distributing an "mlcube.yaml" file. That file can be specified when using the [MLCube runners](https://mlcommons.github.io/mlcube/runners/).
The runner will perform many aspects of configuring your container for you.

Currently, only the mlcube_docker runner is supported.

See the [MLCube documentation](https://mlcommons.github.io/mlcube/) for more details.

[Back To Top &uarr;](#table-of-contents)
