import torch.nn as nn
import sys
import torchvision

def get_model(parameters):
    if parameters["model"]["dimension"] != 2:
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
    parameters["model"]["amp"] = False  # this is not yet implemented for vgg

    return model