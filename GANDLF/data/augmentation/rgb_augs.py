from functools import partial
from torchvision.transforms import ColorJitter
from torchio.transforms import Lambda

def colorjitter(parameters):
    return Lambda(
        function=partial(ColorJitter,
            brightness=parameters["brightness"],
            contrast=parameters["contrast"],
            saturation=parameters["saturation"],
            hue=parameters["hue"],
        ),
        p=parameters["probability"],
    )
