
This file contains mid-level information regarding various parameters that can be leveraged to customize the training/inference in GaNDLF.

## Model

- Defined under the global key `model` in the config file
    - `architecture`: Defines the model architecture (aka "network topology") to be used for training. All options can be found [here](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/models/__init__.py). Some examples are:
        - Segmentation:
            - [Standardized 4-layer UNet](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/models/unet.py) with (`resunet`) and without (`unet`) residual connections, as described in [this paper](https://doi.org/10.1007/978-3-030-46643-5_21).
            - [Multi-layer UNet](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/models/unet_multilayer.py) with (`resunet_multilayer`) and without (`unet_multilayer`) residual connections - this is a more general version of the standard UNet, where the number of layers can be specified by the user.
            - [UNet with Inception Blocks](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/models/uinc.py) (`uinc`) is a variant of UNet with inception blocks, as described in [this paper](https://doi.org/10.48550/arXiv.1907.02110).
            - [UNetR](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/models/unetr.py) (`unetr`) is a variant of UNet with transformers, as described in [this paper](https://doi.org/10.1109/WACV51458.2022.00181).
            - [TransUNet](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/models/transunet.py) (`transunet`) is a variant of UNet with transformers, as described in [this paper](https://doi.org/10.48550/arXiv.2102.04306).
            - And many more.
        - Classification/Regression: 
            - [VGG configurations](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/models/vgg.py) (`vgg11`, `vgg13`, `vgg16`, `vgg19`), as described in [this paper](https://doi.org/10.48550/arXiv.1409.1556). Our implementation allows true 3D computations (as opposed to 2D+1D convolutions).
            - [VGG configurations initialized with weights trained on ImageNet](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/models/imagenet_vgg.py) (`imagenet_vgg11`, `imagenet_vgg13`, `imagenet_vgg16`, `imagenet_vgg19`), as described in [this paper](https://doi.org/10.48550/arXiv.1409.1556).
            - [DenseNet configurations](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/models/densenet.py) (`densenet121`, `densenet161`, `densenet169`, `densenet201`, `densenet264`), as described in [this paper](https://doi.org/10.48550/arXiv.1404.1869). Our implementation allows true 3D computations (as opposed to 2D+1D convolutions).
            - [ResNet configurations](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/models/resnet.py) (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`), as described in [this paper](https://doi.org/10.48550/arXiv.1512.03385). Our implementation allows true 3D computations (as opposed to 2D+1D convolutions).
            - And many more.
    - `dimension`: Defines the dimensionality of convolutions, this is usually the same dimension as the input image, unless specialized processing is done to convert images to a different dimensionality (usually not recommended). For example, 2D images can be stacked to form a "pseudo" 3D image, and 3D images can be processed as "slices" as 2D images.
    - `final_layer`: The final layer of model that will be used to generate the final prediction. Unless otherwise specified, it can be one of `softmax` or `sigmoid` or `logits` or `none` (the latter 2 are only used for regression tasks).
    - `class_list`: The list of classes that will be used for training. This is expected to be a list of integers. 
        - For example, for a segmentation task, this can be a list of integers `[0, 1, 2, 4]` for the BraTS training case for all labels (background, necrosis, edema, and enhancing tumor). Additionally, different labels can be combined to perform "combinatorial training", such as `[0, 1||4, 1||2||4, 4]`, for the BraTS training to train on background, tumor core, whole tumor, and enhancing, respectively.
        - For a classification task, this can be a list of integers `[0, 1]`. 
    - `ignore_label_validation`: This is the location of the label in `class_list` whose performance is to be ignored during metric calculation for validation/testing data
    - `norm_type`: The type of normalization to be used. This can be either `batch` or `instance` or `none`.
    - Various other options specific to architectures, such as (but not limited to):
        - `densenet` models: 
            - `growth_rate`: how many filters to add each layer (k in paper)
            - `bn_size`:  multiplicative factor for number of bottle neck layers # (i.e. bn_size * k features in the bottleneck layer)
            - `drop_rate`: dropout rate after each dense layer
        - `unet_multilayer` and other networks that support multiple layers:
            - `depth`: the number of encoder/decoder (or other types of) layers


## Loss function

- Defined in the `loss_function` parameter of the model configuration.
- By passing `weighted_loss: True`, the loss function will be weighted by the inverse of the class frequency.
- This parameter controls the function which the model is trained. All options can be found [here](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/losses/__init__.py). Some examples are:
    - Segmentation: dice (`dice` or `dc`), dice and cross entropy (`dcce`)
    - Classification/regression: mean squared error (`mse`)
    - And many more.


## Metrics

- Defined in the `metrics` parameter of the model configuration.
- This parameter controls the metrics to be used for model evaluation for the training/validation/testing datasets. All options can be found [here](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/metrics/__init__.py). Most of these metrics are calculated using [TorchMetrics](https://torchmetrics.readthedocs.io/). Some examples are:
    - Segmentation: dice (`dice` and `dice_per_label`), hausdorff distances (`hausdorff` or `hausdorff100` and `hausdorff100_per_label`), hausdorff distances including on the 95th percentile of distances (`hausdorff95` and `hausdorff95_per_label`)  - 
    - Classification/regression: mean squared error (`mse`) calculated per sample
    - Metrics calculated per cohort (these are automatically calculated for classification and regression):
        - Classification: accuracy, precision, recall, f1, for the entire cohort ("global"), per classified class ("per_class"), per classified class averaged ("per_class_average"), per classified class weighted/balanced ("per_class_weighted")
        - Regression: mean absolute error, pearson and spearman coefficients, calculated as mean, sum, or standard.


## Patching Strategy

- `patch_size`: The size of the patch to be used for training. This is expected to be a list of integers, with the length of the list being the same as the dimensionality of the input image. For example, for a 2D image, this can be `[128, 128]`, and for a 3D image, this can be `[128, 128, 128]`.
- `patch_sampler`: The sampler to be used for patch sampling during training. This can be one of `uniform` (the entire input image has equal weight on contributing a valid patch) or `label` (only the regions that have a valid ground truth segmentation label can contribute a patch). `label` sampler usually requires padding of the image to ensure blank patches are not inadvertently sampled; this can be controlled by the `enable_padding` parameter.
- `inference_mechanism`
    - `grid_aggregator_overlap`: this option provides the option to strategize the grid aggregation output; should be either `crop` or `average` - https://torchio.readthedocs.io/patches/patch_inference.html#grid-aggregator
    - `patch_overlap`: the amount of overlap of patches during inference in terms of pixels, defaults to `0`; see https://torchio.readthedocs.io/patches/patch_inference.html#gridsampler for details.


## Data Preprocessing

- Defined in the `data_preprocessing` parameter of the model configuration.
- This parameter controls the various preprocessing functions that are applied to the **entire image** **_before_** the [patching strategy](#patching-strategy) is applied.
- All options can be found [here](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/data/preprocessing/__init__.py). Some of the most important examples are:
    - **Intensity harmonization**: GaNDLF provides multiple normalization and rescaling options to ensure intensity-level harmonization of the entire cohort. Some examples include:
        - `normalize`: simple Z-score normalization
        - `normalize_positive`: this performs z-score normalization only on `pixels > 0`
        - `normalize_nonZero`: this performs z-score normalization only on `pixels != 0`
        - `normalize_nonZero_masked`: this performs z-score normalization only on the region defined by the ground truth annotation
        - `rescale`: simple min-max rescaling, sub-parameters include `in_min_max`, `out_min_max`, `percentiles`; this option is useful to discard outliers in the intensity distribution
        - Template-based normalization: These options take a target image as input (defined by the `target` sub-parameter) and perform different matching strategies to match input image(s) to this target.
            - `histogram_matching`: this performs histogram matching as defined by [this paper](https://doi.org/10.1109/42.836373). 
                - If the `target` image is absent, this will perform global histogram equalization.
                - If `target` is `adaptive`, this will perform [adaptive histogram equalization](https://doi.org/10.1109/83.841534).
            - `stain_normalization`: these are normalization techniques specifically designed for histology images; the different options include `vahadane`, `macenko`, or `ruifrok`, under the `extractor` sub-parameter. Always needs a `target` image to work.
    - **Resolution harmonization**: GaNDLF provides multiple resampling options to ensure resolution-level harmonization of the entire cohort. Some examples include:
        - `resample`: resamples the image to the specified by the `resolution` sub-parameter
        - `resample_min`: resamples the image to the maximum spacing defined by the `resolution` sub-parameter; this is useful in cohorts that have varying resolutions, but the user wants to resample to the minimum resolution for consistency
        - `resize_image`: **NOT RECOMMENDED**; resizes the image to the specified size
        - `resize_patch`: **NOT RECOMMENDED**; resizes the [extracted patch](#patching-strategy) to the specified size
    - And many more.


## Data Augmentation

- Defined in the `data_augmentation` parameter of the model configuration.
- This parameter controls the various augmentation functions that are applied to the **entire image** **_before_** the [patching strategy](#patching-strategy) is applied.
- These should be defined in cognition of the task at hand (for example, RGB augmentations will not work for MRI/CT and other similar radiology images).
- All options can contain a `probability` sub-parameter, which defines the probability of the augmentation being applied to the image. When present, this will supersede the `default_probability` parameter.
- All options can be found [here](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/data/augmentation/__init__.py). Some of the most important examples are:
    - **Radiology-specific augmentations**
        - `kspace`: one of either `ghosting` or `spiking` is picked for augmentation.
        - `bias`: applies a random bias field artefact to the input image using [this function](https://torchio.readthedocs.io/transforms/augmentation.html#randombiasfield).
    - **RGB-specific augmentations**
        - `colorjitter`: applies the [ColorJitter](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html) transform from PyTorch, has sub-parameters `brightness`, `contrast`, `saturation`, and `hue`.
    - **General-purpose augmentations**
        - _Spatial transforms_: they only change the resolution (and thereby, the shape) of the input image, and only apply interpolation to the intensities for consistency
            - `affine`: applies a random affine transformation to the input image; for details, see [this page](https://torchio.readthedocs.io/transforms/augmentation.html#randomaffine); has sub-parameters `scales` (defining the scaling ranges), `degrees` (defining the rotation ranges), and `translation` (defining the translation ranges in **real-world coordinates**, which is usually in _mm_)
            - `elastic`: applies a random elastic deformation to the input image; for details, see [this page](https://torchio.readthedocs.io/transforms/augmentation.html#randomelasticdeformation); has sub-parameters `num_control_points` (defining the number of control points), `locked_borders` (defining the number of locked borders), `max_displacement` (defining the maximum displacement of the control points), `num_control_points` (defining the number of control points), and `locked_borders` (defining the number of locked borders).
            - `flip`: applies a random flip to the input image; for details, see [this page](https://torchio.readthedocs.io/transforms/augmentation.html#randomflip); has sub-parameter `axes` (defining the axes to flip).
            - `rotate`: applies a random rotation by 90 degrees (`rotate_90`) or 180 degrees (`rotate_180`), has sub-parameter `axes` (defining the axes to rotate).
            - `swap`: applies a random swap , has sub-parameter `patch_size` (defining the patch size to swap), and `num_iterations` (number of iterations that 2 patches will be swapped).
        - _Intensity transforms_: they change the intensity of the input image, but never the actual resolution or shape.
            - `motion`: applies a random motion blur to the input image using [this function](https://torchio.readthedocs.io/transforms/augmentation.html#randommotion).
            - `blur`: applies a random Gaussian blur to the input image using [this function](https://torchio.readthedocs.io/transforms/augmentation.html#randomblur)l has sub-parameter `std` (defines the standard deviation range).
            - `noise`: applies a random noise to the input image using [this function](https://torchio.readthedocs.io/transforms/augmentation.html#randomnoise); has sub-parameters `std` (defines the standard deviation range) and `mean` (defines the mean of the noise to be added).
            - `noise_var`: applies a random noise to the input image, however, the with default `std = [0, 0.015 * std(image)]`.
            - `anisotropic`: applies random anisotropic transform to input image using [this function](https://torchio.readthedocs.io/transforms/augmentation.html#randomanisotropy). This changes the resolution and brings it back to its original resolution, thus applying "real-world" interpolation to images.


## Training Parameters

- These are various parameters that control the overall training process.
- `verbose`: generate verbose messages on console; generally used for debugging.
- `batch_size`: defines the batch size to be used for training.
- `in_memory`: this is to enable or disable lazy loading - setting to true reads all data once during data loading, resulting in improvements.
- `num_epochs`: defines the number of epochs to train for.
- `patience`: defines the number of epochs to wait for improvement before early stopping.
- `learning_rate`: defines the learning rate to be used for training.
- `scheduler`: defines the learning rate scheduler to be used for training, more details are [here](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/schedulers/__init__.py); can take the following sub-parameters:
    - `type`: `triangle`, `triangle_modified`, `exp`, `step`, `reduce-on-plateau`, `cosineannealing`, `triangular`, `triangular2`, `exp_range`
    - `min_lr`: defines the minimum learning rate to be used for training.
    - `max_lr`: defines the maximum learning rate to be used for training.
- `optimizer`: defines the optimizer to be used for training, more details are [here](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/optimizers/__init__.py).
- `nested_training`: defines the number of folds to use nested training, takes `testing` and `validation` as sub-parameters, with integer values defining the number of folds to use.
- `memory_save_mode`: if enabled, resize/resample operations in `data_preprocessing` will save files to disk instead of directly getting read into memory as tensors
- **Queue configuration**: this defines how the queue for the input to the model is to be designed **after** the [patching strategy](#patching-strategy) has been applied, and more details are [here](https://torchio.readthedocs.io/data/patch_training.html?#queue). This takes the following sub-parameters:
    - `q_max_length`: his determines the maximum number of patches that can be stored in the queue. Using a large number means that the queue needs to be filled less often, but more CPU memory is needed to store the patches.
    - `q_samples_per_volume`: this determines the number of patches to extract from each volume. A small number of patches ensures a large variability in the queue, but training will be slower.
    - `q_num_workers`: this determines the number subprocesses to use for data loading; '0' means main process is used, scale this according to available CPU resources.
    - `q_verbose`: used to debug the queue
