# Usage

## Example of CLI
```powershell
# continue from previous shell
python GANDLF.py \
  -config ./experiment_0/model.yaml \ # model configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -data ./experiment_0/train.csv \ # data in CSV format 
  -output ./experiment_0/output_dir/ \ # output directory
  -train 1 \ # 1 == train, 0 == inference
  -device 0 # postive integer for GPU device, -1 for CPU
  -modelDir /path/to/model/weights # used in inference mode
```

## Preparing the Data

It is **highly** recommended that the dataset you want to train/infer on has been harmonized:

- Registration
  - Within-modality co-registration [[1](https://doi.org/10.1109/TMI.2014.2377694), [2](https://doi.org/10.1038/sdata.2017.117), [3](https://arxiv.org/abs/1811.02629)]
  - **OPTIONAL**: Registration of all datasets to patient atlas, if applicable [[1](https://doi.org/10.1109/TMI.2014.2377694), [2](https://doi.org/10.1038/sdata.2017.117), [3](https://arxiv.org/abs/1811.02629)]
- Same physical definition of all images (see https://upenn.box.com/v/spacingsIssue for a presentation on how voxel resolutions affects downstream analyses)
- Same intensity profile, i.e., normalization [[4](https://doi.org/10.1016/j.nicl.2014.08.008), [5](https://visualstudiomagazine.com/articles/2020/08/04/ml-data-prep-normalization.aspx), [6](https://developers.google.com/machine-learning/data-prep/transform/normalization), [7](https://towardsdatascience.com/understand-data-normalization-in-machine-learning-8ff3062101f0)]

Recommended tool for tackling all aforementioned preprocessing tasks: https://github.com/CBICA/CaPTk

## Constructing the Data CSV

This application can leverage multiple channels/modalities for training while using a multi-class segmentation file. The expected format is shown as an example in [samples/sample_train.csv](../samples/sample_train.csv) and needs to be structured with the following header format:

```csv
Channel_0,Channel_1,...,Channel_X,Label
/full/path/0.nii.gz,/full/path/1.nii.gz,...,/full/path/X.nii.gz,/full/path/segmentation.nii.gz
```

- `Channel` can be substituted with `Modality` or `Image`
- `Label` can be substituted with `Mask` or `Segmentation`
- Only a single `Label` header should be passed (multiple segmentation classes should be in a single file with unique label numbers)

The [gandlf_constructCSV](../gandlf_constructCSV) can be used to make this easier.

For classification/regression, add a column called `ValueToPredict`. **Note** that currently, we are supporting only a single value prediction per model.

## Customize the Training

All details and comments are in the [samples/sample_training.yaml](../samples/sample_training.yaml).

### Multi-GPU systems

Please ensure that the environment variable `CUDA_VISIBLE_DEVICES` is set [[ref](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/)].

For an example how this is set, see [sge_wrapper](../samples/sge_wrapper).