from pathlib import Path
from .utils import pass_method, map_values
import numpy as np
from skimage.io import imsave


class Patch:
    def __init__(self, slide_path, slide_object, manager, coordinates, level, size, output_suffix="_patch@({},{})_{}x{}.png"):
        self.manager = manager
        self._slide_path = slide_path
        self.slide_object = slide_object
        self.subfolder = slide_path[:slide_path.rindex(".")].split("/")[-1] + "/"
        self.coordinates = coordinates
        self.level = level
        self.size = size
        self.output_suffix = output_suffix

    def read_patch(self):
        return self.slide_object.read_region(
            (self.coordinates[1], self.coordinates[0]), self.level, self.size
        )

    def copy(self):
        return Patch(slide_path=self._slide_path,
                     slide_object=self.slide_object,
                     manager=self.manager,
                     coordinates=self.coordinates,
                     level=self.level,
                     size=self.size)

    def set_slide(self, slide_path):
        self._slide_path = slide_path
        self.subfolder = slide_path[:slide_path.rindex(".")].split("/")[-1] + "/"


    def get_patch_location(self, out_dir):
        path = Path(self._slide_path)

        return out_dir + self.subfolder + path.name.split(path.suffix)[0] + self.output_suffix.format(
            self.coordinates[0], self.coordinates[1], self.size[0], self.size[1])


    def save(self, out_dir, save=True, check_if_valid=True, process_method=None, value_map=None):
        """
        Save patch.
        :param out_dir:
        :return:
        """
        patch = self.read_patch()

        if process_method is None:
            process_method = pass_method

        if check_if_valid:
            for check_function in self.manager.valid_patch_checks:
                if not check_function(patch):
                    return [False, self, ""]
        
        try:
            if save:
                if isinstance(value_map, dict):
                    patch = np.asarray(self.read_patch())[:, :, 0]
                    patch = map_values(patch, value_map)
                    imsave(
                        fname=self.get_patch_location(out_dir),
                        arr=patch
                    )
                elif value_map is None:
                    self.read_patch().save(
                        fp=self.get_patch_location(out_dir),
                        format="PNG"
                    )

            return [True, self, process_method(patch)]

        except Exception as e:
            print(e)
            return [False, self, ""]
