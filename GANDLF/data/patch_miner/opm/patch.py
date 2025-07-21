from pathlib import Path
from .utils import pass_method, map_values
import numpy as np
from skimage.io import imsave
import os

# from pathlib import Path
try:
    from zarr.core import Array
except ImportError as e:
    from zarr.core import array as Array

class Patch:
    def __init__(
        self,
        slide_path: str,
        slide_object: Array,
        manager,
        coordinates,
        level: int,
        size: tuple,
        output_suffix: str = "_patch@{}:{}.png",
    ) -> None:
        """
        Init for Patch.
        @param slide_path: Path to slide. Used primarily for generating patch filenames.
        @param slide_object: OpenSlide object. Read patch images from this object.
        @param manager: PatchManager object. Inheriting this allows this patch to be checked for validity.
        @param coordinates: Ndarray of [x, y] coordinates on slide for the top-left corner of the patch.
        @param level: Level of slide you want to call the patch from.
        @param size: tuple of ints for the side lengths of the patch.
        @param output_suffix: The format to be appended onto the slide's name.
        """
        self.manager = manager
        self._slide_path = slide_path
        self.slide_object = slide_object
        self.subfolder = Path(slide_path).stem
        self.coordinates = coordinates
        self.level = level
        self.size = size
        self.output_suffix = output_suffix

    def read_patch(self):
        """
        Read patch from self.slide_object given this patch's coordinates, level, and size.
        @return: PIL object of RGBA patch image.
        """
        return np.asarray(
            self.slide_object.read_region(
                (self.coordinates[1], self.coordinates[0]), self.level, self.size
            ).convert(
                "RGB"
            )  # openslide-python returns an RGBA PIL image
        )

    def copy(self):
        """
        Return a copy of the current patch.
        @return: new Patch object with same attributes as this one.
        """
        return Patch(
            slide_path=self._slide_path,
            slide_object=self.slide_object,
            manager=self.manager,
            coordinates=self.coordinates,
            level=self.level,
            size=self.size,
        )

    def set_slide(self, slide_path):
        """
        Setter method for changing the slide that this patch belongs to. Useful for pulling corresponding patch from
        multiple slides/label maps.
        @param slide_path: Path of new slide you want to assign this patch to.
        """
        # Change slide path
        self._slide_path = slide_path
        # Re-assign subfolder within output folder
        self.subfolder = Path(slide_path).stem

    def get_patch_path(self, out_dir, create_dir=True):
        """
        Returns string of the path to where this patch will be saved.
        @param out_dir: The output directory
        @return: str
        """
        path = Path(self._slide_path)
        Path(out_dir, self.subfolder).mkdir(parents=True, exist_ok=True)
        return os.path.join(
            out_dir,
            self.subfolder,
            path.name.split(path.suffix)[0]
            + self.output_suffix.format(self.coordinates[0], self.coordinates[1]),
        )

    def save(
        self,
        out_dir,
        label_map=None,
        check_if_valid=True,
        process_method=None,
        value_map=None,
    ):
        """
        Save patch.
        @param out_dir: Output directory for saving the patch. Supplied by patch_manager.py
        @param save: If False it will not write to disk. Helpful for debugging.
        @param check_if_valid: Run through checks supplied by manager. If rejected, don't save.
        @param process_method: A method that takes an image as an input and returns a string. Summarizes patch info.
            If left as None, it will use utils.pass_method() and return an empty string.
        @param value_map: Map key values in patch to alternate value. dict(key => value) where key, value are ints.
            alters the patch by substituting key for value in the image, leaves values not in dictionary unaltered.
            Helpful for standardization.
        @return: A list [bool, Patch, summary]. bool is if the patch was accepted, Patch is the patch object, and
            summary is the output of process_method.
        """

        patch = self.read_patch()

        try:
            imsave(fname=self.get_patch_path(out_dir), arr=patch)
            if process_method is None:
                process_method = pass_method

            if label_map is not None:
                lm_patch = label_map.read_patch()
                if isinstance(value_map, dict):
                    lm_patch = map_values(lm_patch, value_map)
                    imsave(fname=label_map.get_patch_path(out_dir), arr=lm_patch)
                elif value_map is None:
                    imsave(fname=label_map.get_patch_path(out_dir), arr=lm_patch)
            else:
                lm_patch = None


            if check_if_valid:
                for check_function in self.manager.valid_patch_checks:
                    if not check_function(patch):
                        print("Patch failed check", check_function)
                        return [False, self, process_method(patch), lm_patch]

            return [True, self, process_method(patch), lm_patch]

        except Exception as e:
            print("Exception while saving patch:", e)
            return [False, self, "", None]

    def get_data(self, value_map=None):
        """
        Read patch data and optionally apply value mapping.

        Args:
            value_map (dict, optional): Map key values in patch to alternate value.
                Dict of {key: value} pairs where key and value are ints.

        Returns:
            numpy.ndarray: The patch data.
        """
        patch = self.read_patch()

        if isinstance(value_map, dict):
            if len(patch.shape) == 3 and patch.shape[2] >= 3:
                patch = patch[:, :, 0]

            from .utils import map_values
            patch = map_values(patch, value_map)

        return patch

    def validate(self, label_map, process_method=None, value_map=None, check_if_valid=True):
        """
        Validate a patch using the provided check functions.

        Args:
            check_functions (list): List of functions to check patch validity.

        Returns:
            bool: True if the patch passes all checks, False otherwise.
        """
        patch = self.read_patch()

        try:
            if process_method is None:
                process_method = pass_method

            if label_map is not None:
                lm_patch = label_map.read_patch()
                if isinstance(value_map, dict):
                    lm_patch = map_values(lm_patch, value_map)
            else:
                lm_patch = None


            if check_if_valid:
                for check_function in self.manager.valid_patch_checks:
                    if not check_function(patch):
                        print("Patch failed check", check_function)
                        return [False, self, process_method(patch), lm_patch]

            return [True, self, process_method(patch), lm_patch]

        except Exception as e:
            print("Exception while returning patch:", e)
            return [False, self, "", None]
