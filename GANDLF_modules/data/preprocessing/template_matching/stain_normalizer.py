import numpy as np
import torch
import cv2
from torchio.data.image import ScalarImage

from .base import TemplateNormalizeBase
from .utils import rgb2od, od2rgb
from .stain_extractors import VahadaneExtractor, RuifrokExtractor, MacenkoExtractor


class StainNormalizer(TemplateNormalizeBase):
    """Stain normalization base class.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Attributes:
        extractor (CustomExtractor, RuifrokExtractor):
            Method specific stain extractor.
        stain_matrix_target (:class:`numpy.ndarray`):
            Stain matrix of target.
        target_concentrations (:class:`numpy.ndarray`):
            Stain concentration matrix of target.
        maxC_target (:class:`numpy.ndarray`):
            99th percentile of each stain.
        stain_matrix_target_RGB (:class:`numpy.ndarray`):
            Target stain matrix in RGB.

    """

    def __init__(self, extractor="vahadane", **kwargs):
        super().__init__(**kwargs)
        self.stain_matrix_target = None
        self.target_concentrations = None
        self.maxC_target = None
        self.stain_matrix_target_RGB = None

        # we shall always default to vahadane
        self.extractor_str = extractor.lower()
        self.extractor = VahadaneExtractor()
        if self.extractor_str == "ruifrok":
            self.extractor = RuifrokExtractor()
        elif self.extractor_str == "macenko":
            self.extractor = MacenkoExtractor()

    @staticmethod
    def get_concentrations(img, stain_matrix):
        """Estimate concentration matrix given an image and stain matrix.

        Args:
            img (:class:`numpy.ndarray`):
                Input image.
            stain_matrix (:class:`numpy.ndarray`):
                Stain matrix for haematoxylin and eosin stains.

        Returns:
            numpy.ndarray:
                Stain concentrations of input image.

        """
        OD = rgb2od(img).reshape((-1, 3))
        x, _, _, _ = np.linalg.lstsq(stain_matrix.T, OD.T, rcond=-1)
        return x.T

    def fit(self, target: str):
        """Fit to a target image.

        Args:
            target (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
              Target/reference image.

        """
        target_arr = cv2.imread(target)
        target_arr = cv2.cvtColor(target_arr, cv2.COLOR_BGR2RGB).astype(np.uint8)
        self.stain_matrix_target = self.extractor.get_stain_matrix(target_arr)
        self.target_concentrations = self.get_concentrations(
            target_arr, self.stain_matrix_target
        )
        self.maxC_target = np.percentile(
            self.target_concentrations, 99, axis=0
        ).reshape((1, 2))
        # useful to visualize.
        self.stain_matrix_target_RGB = od2rgb(self.stain_matrix_target)

    def apply_normalize(self, image: ScalarImage) -> None:
        image_transformed = torch.from_numpy(self.transform(image.data.numpy()))
        image.set_data(image_transformed.unsqueeze(0))

    def transform(self, img):
        """Transform an image.

        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                RGB input source image.

        Returns:
            :class:`numpy.ndarray`:
                RGB stain normalized image.

        """
        if self.maxC_target is None:
            self.fit(self.target)

        # ensure image is in the format opencv would expect
        if img.shape[-1] == 1:
            img = np.squeeze(img, -1)
        if img.shape[0] == 3:
            img = img.transpose((1, 2, 0))

        stain_matrix_source = self.extractor.get_stain_matrix(img)
        source_concentrations = self.get_concentrations(img, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= self.maxC_target / maxC_source
        trans = 255 * np.exp(
            -1 * np.dot(source_concentrations, self.stain_matrix_target)
        )

        # ensure between 0 and 255
        trans[trans > 255] = 255
        trans[trans < 0] = 0

        return trans.reshape(img.shape).astype(np.uint8)


def stain_normalizer(parameters):
    """
    This function is a wrapper for histogram matching.

    Args:
        parameters (dict): Dictionary of parameters.
    """
    extractor = parameters.get("extractor", "ruifrok")
    target = parameters.get("target", None)
    if target is None:
        raise ValueError("Target image is required.")

    return StainNormalizer(target=target, extractor=extractor)
