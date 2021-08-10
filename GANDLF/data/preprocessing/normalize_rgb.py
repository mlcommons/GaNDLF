from torchvision.transforms import Normalize


def normalize_by_val(input_tensor, mean, std):
    """
    This function returns the tensor normalized by these particular values
    """
    normalizer = Normalize(mean, std)
    return normalizer(input_tensor)


def normalize_imagenet(input_tensor):
    """
    This function returns the tensor normalized by standard imagenet values
    """
    return normalize_by_val(
        input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )


def normalize_standardize(input_tensor):
    """
    This function returns the tensor normalized by subtracting 128 and dividing by 128
    image = (image - 128)/128
    """
    return normalize_by_val(input_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def normalize_div_by_255(input_tensor):
    """
    This function divides all values of the input tensor by 255 on all channels
    image = image/255
    """
    return normalize_by_val(input_tensor, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
