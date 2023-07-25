import torch.nn as nn
import sys
import torchvision


def brainage(parameters):
    """
    This function creates a VGG16-based neural network model for brain age prediction.

    Args:
        parameters (dict): A dictionary containing the model parameters, including:
            - model: a sub-dictionary containing the model-specific parameters.
            - dimension: the dimensionality of the input data (should be 2 for brain age prediction).
            - amp: whether to use automatic mixed precision (not yet implemented for VGG).

    Returns:
        model (torch.nn.Module): A VGG16-based neural network model with a modified classifier layer for brain age prediction.
    """

    # Check that the input data is 2D
    if parameters["model"]["dimension"] != 2:
        sys.exit("Brain Age predictions only works on 2D data")

    # Load the pretrained VGG16 model
    model = torchvision.models.vgg16(pretrained=True)

    # Remove the final convolutional layer
    model.final_convolution_layer = None

    # Freeze the parameters of all layers in the feature extraction part of the model
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify the classifier layer for brain age prediction
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove the last layer
    features.extend(
        [
            nn.Linear(
                num_features, 1024
            ),  # Add a linear layer with 1024 output features
            nn.ReLU(True),  # Add a ReLU activation function
            nn.Dropout2d(0.8),  # Add a 2D dropout layer with a probability of 0.8
            nn.Linear(
                1024, 1
            ),  # Add a linear layer with 1 output feature (for brain age prediction)
        ]
    )
    model.classifier = nn.Sequential(
        *features
    )  # Replace the model classifier with the modified one

    # Set the "amp" parameter to False (not yet implemented for VGG)
    parameters["model"]["amp"] = False

    return model
