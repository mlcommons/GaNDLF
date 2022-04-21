from torchio.transforms.intensity_transform import IntensityTransform
from torchio.data.subject import Subject
from torchio.data.image import ScalarImage


class TemplateNormalizeBase(IntensityTransform):
    """The Base class to apply template-based normalization techniques.

    Args:
        target (str): The target image on the basis of which the subject's images are to be normalized.
    """

    def __init__(self, target: str = None, **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.args_names = ("target",)

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            self.apply_normalize(image)
        return subject

    def apply_normalize(self, image: ScalarImage) -> None:
        # ensure you call image.set_data() with the normalized tensor/array/image
        raise NotImplementedError("This method must be implemented in a subclass.")
