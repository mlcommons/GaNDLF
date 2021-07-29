import sys
import torch.optim as optim
from torch.optim.lr_scheduler import *
from GANDLF.schd import *
from GANDLF.models.fcn import fcn
from GANDLF.models.unet import unet
from GANDLF.models.uinc import uinc
from GANDLF.models.MSDNet import MSDNet
from GANDLF.models.sdnet import SDNet
from GANDLF.models import densenet
from GANDLF.models.vgg import VGG, make_layers, cfg
from GANDLF.losses import *
from GANDLF.utils import *
from GANDLF.metrics import fetch_metric
import torchvision
import torch.nn as nn


def get_model(
    modelname,
    num_dimensions,
    num_channels,
    num_classes,
    base_filters,
    norm_type,
    final_convolution_layer,
    patch_size,
    batch_size,
    **kwargs
):
    """
    This function takes the default constructor and returns the model

    kwargs can be used to pass key word arguments and use arguments that are not explicitly defined.
    """

    divisibilityCheck_patch = True
    divisibilityCheck_baseFilter = True

    divisibilityCheck_denom_patch = 16  # for unet/resunet/uinc
    divisibilityCheck_denom_baseFilter = 4  # for uinc

    if "amp" in kwargs:
        amp = kwargs.get("amp")
    else:
        amp = False
    if modelname == "resunet":
        model = unet(
            num_dimensions,
            num_channels,
            num_classes,
            base_filters,
            norm_type,
            final_convolution_layer=final_convolution_layer,
            residualConnections=True,
        )
        divisibilityCheck_baseFilter = False

    elif modelname == "unet":
        model = unet(
            num_dimensions,
            num_channels,
            num_classes,
            base_filters,
            norm_type,
            final_convolution_layer=final_convolution_layer,
        )
        divisibilityCheck_baseFilter = False

    # elif modelname == "light_resunet":
    #     model = light_unet(
    #         num_dimensions,
    #         num_channels,
    #         num_classes,
    #         base_filters,
    #         norm_type,
    #         final_convolution_layer=final_convolution_layer,
    #         residualConnections=True,
    #     )
    #     divisibilityCheck_baseFilter = False

    # elif modelname == "light_unet":
    #     model = light_unet(
    #         num_dimensions,
    #         num_channels,
    #         num_classes,
    #         base_filters,
    #         norm_type,
    #         final_convolution_layer=final_convolution_layer,
    #     )
    #     divisibilityCheck_baseFilter = False

    elif modelname == "fcn":
        model = fcn(
            num_dimensions,
            num_channels,
            num_classes,
            base_filters,
            norm_type,
            final_convolution_layer=final_convolution_layer,
        )
        # not enough information to perform checking for this, yet
        divisibilityCheck_patch = False
        divisibilityCheck_baseFilter = False

    elif modelname == "uinc":
        model = uinc(
            num_dimensions,
            num_channels,
            num_classes,
            base_filters,
            norm_type,
            final_convolution_layer=final_convolution_layer,
        )

    elif modelname == "msdnet":
        model = MSDNet(
            num_dimensions,
            num_channels,
            num_classes,
            base_filters,
            norm_type,
            final_convolution_layer=final_convolution_layer,
        )
        amp = False  # this is not yet implemented for msdnet

    elif modelname == "sdnet":

        if patch_size != [224, 224, 1]:
            print(
                "WARNING: The patch size is not 224x224, which is required for sdnet. Using default patch size instead",
                file=sys.stderr,
            )
            patch_size = [224, 224, 1]

        if batch_size == 1:
            raise ValueError("'batch_size' needs to be greater than 1 for 'sdnet'")

        model = SDNet(
            num_dimensions,
            num_channels,
            num_classes,
            base_filters,
            norm_type,
            final_convolution_layer=final_convolution_layer,
        )
        amp = False  # this is not yet implemented for sdnet

    # these are generic imagenet-trained models and should be customized
    elif "imagenet" in modelname:

        if num_dimensions != 2:
            sys.exit("ImageNet-trained models only work on 2D data")

        divisibilityCheck_patch = False
        divisibilityCheck_baseFilter = False

        if "batch_norm" in kwargs:
            batch_norm = kwargs.get("batch_norm")
        else:
            batch_norm = True

        if "vgg11" in modelname:
            if batch_norm:
                model = torchvision.models.vgg11_bn(pretrained=True)
            else:
                model = torchvision.models.vgg11(pretrained=True)
        elif "vgg13" in modelname:
            if batch_norm:
                model = torchvision.models.vgg13_bn(pretrained=True)
            else:
                model = torchvision.models.vgg13(pretrained=True)
        elif "vgg16" in modelname:
            if batch_norm:
                model = torchvision.models.vgg16_bn(pretrained=True)
            else:
                model = torchvision.models.vgg16(pretrained=True)
        elif "vgg19" in modelname:
            if batch_norm:
                model = torchvision.models.vgg19_bn(pretrained=True)
            else:
                model = torchvision.models.vgg19(pretrained=True)
        elif "squeezenet1_0" in modelname:
            model = torchvision.models.squeezenet1_0(pretrained=True)
        elif "squeezenet1_1" in modelname:
            model = torchvision.models.squeezenet1_1(pretrained=True)
        elif "inceptionv3" in modelname:
            model = torchvision.models.inception_v3(pretrained=True)
        elif "densenet121" in modelname:
            model = torchvision.models.densenet121(pretrained=True)
        elif "densenet161" in modelname:
            model = torchvision.models.densenet161(pretrained=True)
        elif "densenet169" in modelname:
            model = torchvision.models.densenet169(pretrained=True)
        elif "densenet201" in modelname:
            model = torchvision.models.densenet201(pretrained=True)
        elif "densenet264" in modelname:
            model = torchvision.models.densenet264(pretrained=True)
        elif "resnet18" in modelname:
            model = torchvision.models.resnet18(pretrained=True)
        elif "resnet34" in modelname:
            model = torchvision.models.resnet34(pretrained=True)
        elif "resnet50" in modelname:
            model = torchvision.models.resnet50(pretrained=True)
        elif "resnet101" in modelname:
            model = torchvision.models.resnet101(pretrained=True)
        elif "resnet152" in modelname:
            model = torchvision.models.resnet152(pretrained=True)
        else:
            sys.exit(
                "Could not find the requested model '"
                + modelname
                + "' in the implementation"
            )

    elif "densenet" in modelname:
        if modelname == "densenet121":  # regressor/classifier network
            model = densenet.generate_model(
                model_depth=121,
                num_classes=num_classes,
                num_dimensions=num_dimensions,
                num_channels=num_channels,
                final_convolution_layer=final_convolution_layer,
            )
        elif modelname == "densenet161":  # regressor/classifier network
            model = densenet.generate_model(
                model_depth=161,
                num_classes=num_classes,
                num_dimensions=num_dimensions,
                num_channels=num_channels,
                final_convolution_layer=final_convolution_layer,
            )
        elif modelname == "densenet169":  # regressor/classifier network
            model = densenet.generate_model(
                model_depth=169,
                num_classes=num_classes,
                num_dimensions=num_dimensions,
                num_channels=num_channels,
                final_convolution_layer=final_convolution_layer,
            )
        elif modelname == "densenet201":  # regressor/classifier network
            model = densenet.generate_model(
                model_depth=201,
                num_classes=num_classes,
                num_dimensions=num_dimensions,
                num_channels=num_channels,
                final_convolution_layer=final_convolution_layer,
            )
        elif modelname == "densenet264":  # regressor/classifier network
            model = densenet.generate_model(
                model_depth=264,
                num_classes=num_classes,
                num_dimensions=num_dimensions,
                num_channels=num_channels,
                final_convolution_layer=final_convolution_layer,
            )
        else:
            sys.exit(
                "Requested DENSENET type '" + modelname + "' has not been implemented"
            )

        amp = False  # this is not yet implemented for densenet
        # these checks are not needed
        divisibilityCheck_patch = False
        divisibilityCheck_baseFilter = False

    elif "vgg" in modelname:  # common parsing for vgg
        if modelname == "vgg11":
            vgg_config = cfg["A"]
        elif modelname == "vgg13":
            vgg_config = cfg["B"]
        elif modelname == "vgg16":
            vgg_config = cfg["D"]
        elif modelname == "vgg19":
            vgg_config = cfg["E"]
        else:
            sys.exit("Requested VGG type '" + modelname + "' has not been implemented")

        amp = False  # this is not yet implemented for vgg
        # these checks are not needed
        divisibilityCheck_patch = False
        divisibilityCheck_baseFilter = False

        if "batch_norm" in kwargs:
            batch_norm = kwargs.get("batch_norm")
        else:
            batch_norm = True
        layers = make_layers(
            vgg_config, num_dimensions, num_channels, batch_norm=batch_norm
        )
        # num_classes is coming from 'class_list' in config, which needs to be changed to use a different variable for regression
        model = VGG(
            num_dimensions,
            layers,
            num_classes,
            final_convolution_layer=final_convolution_layer,
        )

    elif modelname == "brain_age":
        if num_dimensions != 2:
            sys.exit("Brain Age predictions only works on 2D data")
        model = torchvision.models.vgg16(pretrained=True)
        model.final_convolution_layer = None
        # Freeze training for all layers
        for param in model.features.parameters():
            param.require_grad = False
        # Newly created modules have require_grad=True by default
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]  # Remove last layer
        # features.extend([nn.AvgPool2d(1024), nn.Linear(num_features,1024),nn.ReLU(True), nn.Dropout2d(0.8), nn.Linear(1024,1)]) # RuntimeError: non-empty 2D or 3D (batch mode) tensor expected for input
        features.extend(
            [
                nn.Linear(num_features, 1024),
                nn.ReLU(True),
                nn.Dropout2d(0.8),
                nn.Linear(1024, 1),
            ]
        )
        model.classifier = nn.Sequential(*features)  # Replace the model classifier
        amp = False  # this is not yet implemented for vgg
        # these checks are not needed
        divisibilityCheck_patch = False
        divisibilityCheck_baseFilter = False

    else:
        raise ValueError(
            "WARNING: Could not find the requested model '"
            + modelname
            + "' in the implementation"
        )

    # check divisibility
    if divisibilityCheck_patch:
        if not checkPatchDivisibility(patch_size, divisibilityCheck_denom_patch):
            sys.exit(
                "The 'patch_size' should be divisible by '"
                + str(divisibilityCheck_denom_patch)
                + "' for the '"
                + modelname
                + "' architecture"
            )
    if divisibilityCheck_baseFilter:
        if base_filters % divisibilityCheck_denom_baseFilter != 0:
            sys.exit(
                "The 'base_filters' should be divisible by '"
                + str(divisibilityCheck_denom_baseFilter)
                + "' for the '"
                + modelname
                + "' architecture"
            )

    return model, amp


def get_optimizer(optimizer_name, model, learning_rate):
    """
    This function parses the optimizer from the config file and returns the appropriate object
    """
    model_parameters = model.parameters()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            model_parameters, lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.00005
        )
    else:
        print(
            "WARNING: Could not find the requested optimizer '"
            + optimizer_name
            + "' in the implementation, using sgd, instead",
            file=sys.stderr,
        )
        optimizer = optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)

    return optimizer


def get_scheduler(
    which_scheduler, optimizer, batch_size, training_samples_size, learning_rate
):
    """
    This function parses the optimizer from the config file and returns the appropriate object
    """
    step_size = 4 * batch_size * training_samples_size
    if which_scheduler == "triangle":
        clr = cyclical_lr(step_size, min_lr=10 ** -3, max_lr=1)
        scheduler_lr = LambdaLR(optimizer, [clr])
        print("Initial Learning Rate: ", learning_rate)
    elif which_scheduler == "triangle_modified":
        step_size = training_samples_size / learning_rate
        clr = cyclical_lr_modified(step_size)
        scheduler_lr = LambdaLR(optimizer, [clr])
        print("Initial Learning Rate: ", learning_rate)
    elif which_scheduler == "exp":
        scheduler_lr = ExponentialLR(optimizer, learning_rate, last_epoch=-1)
    elif which_scheduler == "step":
        scheduler_lr = StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    elif which_scheduler == "reduce-on-plateau":
        scheduler_lr = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        )
    elif which_scheduler == "triangular":
        scheduler_lr = CyclicLR(
            optimizer,
            learning_rate * 0.001,
            learning_rate,
            step_size_up=step_size,
            step_size_down=None,
            mode="triangular",
            gamma=1.0,
            scale_fn=None,
            scale_mode="cycle",
            cycle_momentum=False,
            base_momentum=0.8,
            max_momentum=0.9,
            last_epoch=-1,
        )
    elif which_scheduler == "triangular2":
        scheduler_lr = CyclicLR(
            optimizer,
            learning_rate * 0.001,
            learning_rate,
            step_size_up=step_size,
            step_size_down=None,
            mode="triangular2",
            gamma=1.0,
            scale_fn=None,
            scale_mode="cycle",
            cycle_momentum=False,
            base_momentum=0.8,
            max_momentum=0.9,
            last_epoch=-1,
        )
    elif which_scheduler == "exp_range":
        scheduler_lr = CyclicLR(
            optimizer,
            learning_rate * 0.001,
            learning_rate,
            step_size_up=step_size,
            step_size_down=None,
            mode="exp_range",
            gamma=1.0,
            scale_fn=None,
            scale_mode="cycle",
            cycle_momentum=False,
            base_momentum=0.8,
            max_momentum=0.9,
            last_epoch=-1,
        )
    elif which_scheduler == "cosineannealing":
        scheduler_lr = CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
    else:
        print(
            "WARNING: Could not find the requested Learning Rate scheduler '"
            + which_scheduler
            + "' in the implementation, using exp, instead",
            file=sys.stderr,
        )
        scheduler_lr = ExponentialLR(optimizer, 0.1, last_epoch=-1)

    return scheduler_lr


def get_loss_and_metrics(image, ground_truth, predicted, params):
    """
    image: torch.Tensor
        The input image stack according to requirements
    ground_truth : torch.Tensor
        The input ground truth for the corresponding image label
    predicted : torch.Tensor
        The input predicted label for the corresponding image label
    params : dict
        The parameters passed by the user yaml

    Returns
    -------
    loss : torch.Tensor
        The computed loss from the label and the output
    metric_output : torch.Tensor
        The computed metric from the label and the output
    """
    loss_function = fetch_loss_function(params["loss_function"], params)
    if len(predicted) > 1:
        loss_seg = loss_function(predicted[0], ground_truth.squeeze(-1), params)
        loss_function = fetch_loss_function("l1", None)
        loss_reco = loss_function(predicted[1], image[:, :1, ...], None)
        loss_function = fetch_loss_function("kld", params)
        loss_kld = loss_function(predicted[2], predicted[3])
        loss_function = fetch_loss_function("mse", None)
        loss_cycle = loss_function(predicted[2], predicted[4], None)
        loss = 0.01 * loss_kld + loss_reco + 10 * loss_seg + loss_cycle
    else:
        loss = loss_function(predicted, ground_truth, params)
    metric_output = {}
    # Metrics should be a list
    for metric in params["metrics"]:
        metric_function = fetch_metric(metric)  # Write a fetch_metric
        if len(predicted) > 1:
            metric_output[metric] = (
                metric_function(predicted[0], ground_truth.squeeze(-1), params)
                .cpu()
                .data.item()
            )
        else:
            metric_output[metric] = (
                metric_function(predicted, ground_truth, params).cpu().data.item()
            )
    return loss, metric_output
