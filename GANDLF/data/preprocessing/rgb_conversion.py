import torch

import PIL.Image
import numpy as np

from torchio.transforms.intensity_transform import IntensityTransform
from torchio.data.subject import Subject


class RGB2RGBA(IntensityTransform):
    """
    Convert RGB image to RGBA image.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            image_data_to_set = image.data
            # only proceed with RGB is detected
            if image_data_to_set.shape[0] == 3:
                image_data_array = image_data_to_set.numpy()
                if image_data_array.shape[-1] == 1:
                    image_data_array = image_data_array[..., 0]
                    image_data_array = image_data_array.transpose([1, 2, 0])
                image_pil = PIL.Image.fromarray(image_data_array.astype(np.uint8))
                image_pil_rgb = image_pil.convert("RGBA")
                image_data_to_set = torch.from_numpy(
                    np.array(image_pil_rgb).transpose([2, 0, 1])
                ).unsqueeze(-1)
            image.set_data(image_data_to_set)
        return subject


class RGBA2RGB(IntensityTransform):
    """
    Convert RGBA image to RGB image.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            image_data_to_set = image.data
            # only proceed with RGBA is detected
            if image_data_to_set.shape[0] == 4:
                image_data_array = image_data_to_set.numpy()
                if image_data_array.shape[-1] == 1:
                    image_data_array = image_data_array[..., 0]
                    image_data_array = image_data_array.transpose([1, 2, 0])
                image_pil = PIL.Image.fromarray(image_data_array.astype(np.uint8))
                image_pil_rgb = image_pil.convert("RGB")
                image_data_to_set = torch.from_numpy(
                    np.array(image_pil_rgb).transpose([2, 0, 1])
                ).unsqueeze(-1)
            image.set_data(image_data_to_set)
        return subject


def rgba2rgb_transform(parameters=None):
    return RGBA2RGB()


def rgb2rgba_transform(parameters=None):
    return RGB2RGBA()
