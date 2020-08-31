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

## Constructing the Data CSV

This application can leverage multiple channels/modalities for training while using a multi-class segmentation file. The expected format is shown as an example in [samples/sample_train.csv](../samples/sample_train.csv) and needs to be structured with the following header format:

```csv
Channel_0,Channel_1,...,Channel_X,Label
/full/path/0.nii.gz,/full/path/1.nii.gz,...,/full/path/X.nii.gz,/full/path/segmentation.nii.gz
```

- `Channel` can be substituted with `Modality` or `Image`
- `Label` can be substituted with `Mask` or `Segmentation`
- Only a single `Label` header should be passed (multiple segmentation classes should be in a single file with unique label numbers)

## Customize the Training

All details and comments are in the [samples/sample_training.yaml](../samples/sample_training.yaml).

### Multi-GPU systems

Please ensure that the environment variable `CUDA_VISIBLE_DEVICES` is set [ref](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/).
