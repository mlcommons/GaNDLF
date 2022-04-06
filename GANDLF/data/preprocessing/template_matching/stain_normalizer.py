import numpy as np
import torch
import SimpleITK as sitk
from torchio.data.image import ScalarImage

from .base import TemplateNormalizeBase
from .utils import rgb2od, od2rgb

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
        self.extractor = extractor.lower()
        self.stain_matrix_target = None
        self.target_concentrations = None
        self.maxC_target = None
        self.stain_matrix_target_RGB = None

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
        target_arr = sitk.GetArrayFromImage(sitk.ReadImage(target))
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