# GANDLF Models

## Current Models

|            **Architecture**           |                   **Link**                  |
|:-------------------------------------:|:-------------------------------------------:|
| [Brain Age Predictor](./brain_age.py) |    https://doi.org/10.1093/brain/awaa160    |
|      [Deep UNet](./deep_unet.py)      | https://doi.org/10.1109/JSTARS.2018.2833382 |
|       [DenseNet](./densenet.py)       |       https://arxiv.org/abs/2004.04968      |
|   [EfficientNet](./efficientnet.py)   |       https://arxiv.org/abs/1905.11946      |
|  [Fully Connected Network](./fcn.py)  |       https://arxiv.org/abs/1411.4038       |
|         [ResNet](./resnet.py)         |     https://arxiv.org/pdf/1512.03385.pdf    |
|          [SDNet](./sdnet.py)          | https://doi.org/10.1016/j.media.2019.101535 |
|           [UInc](./uinc.py)           |       https://arxiv.org/abs/1907.02110      |
|           [Unet](./unet.py)           |      https://arxiv.org/abs/1606.06650v1     |
|            [VGG](./vgg.py)            |       https://arxiv.org/abs/1409.1556       | 

## Adding a new model

- Follow example of `GANDLF.models.unet`.
- Define a new submodule under `GANDLF.models`, and define a class that inherits from `GANDLF.models.ModelBase`.
- Ensure that a forward pass is implemented.
- All parameters should be taken as input, with special parameters (for e.g., `residualConnections` for `unet`) should not be exposed to the parameters dict, and should be handled separately via another class.
    - For example, `GANDLF.models.unet.unet` has a `residualConnections` parameter, which is not exposed to the parameters dict, and a separate class `GANDLF.models.unet.resunet` is defined which enables this flag.
- Add the model's identifier to `GANDLF.models.__init__.global_model_dict` as appropriate.
- Call the new mode from the config using the `model` key.