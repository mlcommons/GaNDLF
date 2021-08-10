from functools import partial
from torchio.transforms import (
    Normalize,
    Lambda,
)


## define the functions that need to wrapped into lambdas
def normalize_by_val(input_tensor, mean, std):
    """
    This function returns the tensor normalized by these particular values
    """
    normalizer = Normalize(mean, std)
    return normalizer(input_tensor)


# the "_transform" functions return lambdas that can be used to wrap into a Compose class
def normalize_by_val_transform(mean, std, p=1):
    return Lambda(
        function=partial(normalize_by_val, mean=mean, std=std),
        p=p,
    )


def normalize_imagenet_transform(p=1):
    return Lambda(
        function=partial(
            normalize_by_val, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        p=p,
    )


def normalize_standardize_transform(p=1):
    return Lambda(
        function=partial(normalize_by_val, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        p=p,
    )


def normalize_div_by_255_transform(p=1):
    return Lambda(
        function=partial(normalize_by_val, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        p=p,
    )
