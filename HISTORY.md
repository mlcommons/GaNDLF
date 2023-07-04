## 0.0.17
- Added a CLI for metrics computation
- Added metrics specific for image-to-image comparison
- Allow `penalty` and `class_weights` defined in the config to be used instead of a re-computation
- Resume functionality now includes penalty calculation

## 0.0.16
- Added a script "gandlf_deploy", allowing deployment of models into MLCubes (currently requires Docker)
- ImageNet pre-trained models for UNet with variable encoders is now available
- ACS/Soft conversion is available for ImageNet-pretrained UNet
- Updated links, copyright and email to MLCommons
- Allowing provision for user to generate multiple configurations for experimentation
- Added ability to combine classification inference results from different architectures
- Added ability to save `initial` and `latest` models in addition to `best`
- Added ability to specify testing data csv in main cli
- Normalized surface dice has been added
- Added dedicated script to perform post-training model optimization
- Added CI and documentation for OpenFL integration
- Added getting started guide
- Added documentation for all loss functions and updated guideline
- Added ability to save resized images instead of loading them directly

## 0.0.15
- Updated `setup.py` for `python>=3.8`
- `stride_size` is now handled internally for histology data
- Probability maps are now saved overlaid with original WSI
- Added ability to print model size and summary at run-time
- Improved error checking added for WSI inference
- VIPS has been removed from dependencies
- Failed unit test cases are now recorded
- Per class accuracy has been added as a metric
- Dedicated rescaling preprocessing function added for increased flexibility
- Largest Connected Component Analysis is now added
- Included metrics using overall predictions and ground truths

## 0.0.14

- Add an option (`"save_training"`) to save training patches
- Add option to save per-label segmentation metrics
- Separate `"motion"` artifact
- `DenseNet` now supports `InstanceNorm`
- Updated implementations of `VGG` and `DenseNet` to use `ModelBase` for consistency
- Model saving now includes the git commit hash
- Added FAQ in documentation
- Accuracy is now standardized from `torchmetrics`
- New post-processing module added
- Anonymization module has been added
- More progress bars added for better feedback
- NIfTI conversion added in anonymization
- Using TiffSlide instead of OpenSlide
- Minimum resampling resolution is now available
- Adding option to resize images and resize patches separately
- Reverse one-hot logic is now updated to output unique labels
- User can now resume previous training with and without parameter/data updates
- Docker images are now getting built
- Inference works without having access to ground truth labels
- Map output labels using post-processing before saving
- Enable customized histology classification output via heatmaps
- Added ImageNet pre-trained models
- Added RGBA to RGB conversion for preprocessing
- Model can now be saved at every epoch
- Different options for final inference
- Added submodule to handle template-based normalization
- RGB conversion submodule added to handle alpha channel conversions
- Sigmoid multiplier option has been added
- Compute objects can now be requested using developer-level functions
- Transformer-based networks, TransUNet and UNetR are now available
- Can now perform histology computation using microns

## 0.0.13

- Deep supervision added
- Documentation updated
- Model IO is now standardized

## 0.0.12

- Misc bugfixes
- Automatic check-pointing of the model has been added
- Extending the codebase has been simplified
- New optimizers added
- New metrics added
- Affine augmentation can now be significantly fine-tuned
- Update logic for penalty calculation
- RGB-specific augmentation added
- Cropping added

## 0.0.11

- Misc bugfixes for segmentation and classification
- DFU 2021 parameter file added
- Added SDNet for supervised learning - https://doi.org/10.1016/j.media.2019.101535
- Added option to re-orient all images to canonical
- Preprocessing and augmentation made into separate submodules

## 0.0.10

- Half-time epoch loss and metric output added for increased information
- Gradient clipping added
- Per-epoch details in validation output added
- Different types of normalization layer options added
- Hausdorff as a validation metric has been added
- New option to save preprocessed data before the training starts

## 0.0.9

- Refactoring the training and inference code
- Added offline mechanism to generate padded images to improve training RAM requirements

## 0.0.8

- Pre-split training/validation data can now be provided
- Major code refactoring to make extensions easier
- Added a way to ignore a label during validation dice calculation
- Added more options for VGG
- Tests can now be run on GPU
- New scheduling options added

## 0.0.7

- New modality switch added for rad/path
- Class list can now be defined as a range
- Added option to train and infer on fused labels
- Rotation 90 and 180 augmentation added
- Cropping zero planes added for preprocessing
- Normalization options added
- Added option to save generated masks on validation and (if applicable) testing data

## 0.0.6

- Added PyVIPS support
- SubjectID-based split added

## 0.0.5

- 2D support added
- Pre-processing module added
  - Added option to threshold or clip the input image
- Code consolidation
- Added generic DenseNet
- Added option to switch between Uniform and Label samplers
- Added histopathology input (patch-based extraction)

## 0.0.4

- Added full image validation for generating loss and dice scores
- Nested cross-validation added
  - Collect statistics and plot them
- Weighted DICE computation for handling class imbalances in segmentation

## 0.0.3 

- Added detailed documentation
- Added MSE from Torch 
- Added option to parameterize model properties
  - Final convolution layer (softmax/sigmoid/none)
- Added option to resize input dataset
- Added new regression architecture (VGG)
- Version checking in config file

## 0.0.2

- More scheduling options
- Automatic mixed precision training is now enabled by default
- Subject-based shuffle for training queue construction is now enabled by default
- Single place to parse and pass around parameters to make training/inference API easier to handle
- Configuration file mechanism switched to YAML

## 0.0.1 (2020/08/25)

- First tag of GaNDLF
- Initial feature list:
  - Supports multiple
    - Deep Learning model architectures
    - Channels/modalities 
    - Prediction classes
  - Data augmentation
  - Built-in cross validation
