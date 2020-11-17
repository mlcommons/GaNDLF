import concurrent.futures
import os
from functools import partial
from .patch import Patch
from .config import PATCH_SIZE, READ_TYPE, OVERLAP_FACTOR, VALUE_MAP
import numpy as np
import openslide
from tqdm import tqdm
from .utils import get_nonzero_percent


class PatchManager:
    """Manager for patches for openslide"""

    def __init__(self, filename):
        """
        Initialize
        :param filename: path to openslide supported file (str)
        """
        self.patches = list()
        self.path = filename
        self.slide_object = openslide.open_slide(filename)
        self.slide_dims = openslide.open_slide(self.path).dimensions
        self.slide_subfolder = filename[:filename.rindex(".")].split("/")[-1] + "/"
        self.valid_mask = None
        self.mined_mask = None
        self.valid_mask_scale = (0, 0)
        self.valid_patch_checks = []
        self.label_map = None
        self.label_map_object = None
        self.label_map_subfolder = None
        self.label_map_patches = list()

    def set_slide_path(self, filename):
        """
        Set path for slide
        :param filename: path to openslide supported file (str)
        :return: None
        """
        self.path = filename

    def set_label_map(self, path):
        self.label_map = path
        self.label_map_object = openslide.open_slide(path)
        self.label_map_subfolder = path[:path.rindex(".")].split("/")[-1] + "/"

    def set_valid_mask(self, mask, scale=(1, 1)):
        self.valid_mask = mask
        self.mined_mask = np.zeros_like(mask)
        self.valid_mask_scale = scale

    def add_patch(self, patch):
        """
        Add patch to manager
        :param patch: Patch object to add to set of patches
        :return: None
        TODO: test hashing to ensure same patch won't be added
        """
        try:
            if OVERLAP_FACTOR != 1 and self.valid_mask is None:
                print("OVERLAP_FACTOR can only be not one if valid_mask is set.")
                exit(1)
            inverse_overlap_factor = 1 - OVERLAP_FACTOR
            valid_start_x = int(round(
                (patch.coordinates[0] - int(round((PATCH_SIZE[0] + 1) * inverse_overlap_factor))) /
                self.valid_mask_scale[0]))
            valid_start_y = int(round(
                (patch.coordinates[1] - int(round((PATCH_SIZE[1] + 1) * inverse_overlap_factor))) /
                self.valid_mask_scale[1]))
            if OVERLAP_FACTOR != 1:
                valid_end_x = int(round(
                    (patch.coordinates[0] + int(round(PATCH_SIZE[0] * inverse_overlap_factor))) / self.valid_mask_scale[
                        0]))
                valid_end_y = int(round(
                    (patch.coordinates[1] + int(round(PATCH_SIZE[1] * inverse_overlap_factor))) / self.valid_mask_scale[
                        1]))
                self.valid_mask[
                self.min_bound_check(valid_start_x):self.width_bound_check(valid_end_x),
                self.min_bound_check(valid_start_y):self.height_bound_check(valid_end_y)
                ] = False
            else:
                self.valid_mask[
                    valid_start_x, valid_start_y] = False  # Change only the starting index to prevent calling the same patch

            mined_start_x = int(round((patch.coordinates[0]) / self.valid_mask_scale[0]))
            mined_start_y = int(round((patch.coordinates[1]) / self.valid_mask_scale[1]))
            mined_end_x = int(round((patch.coordinates[0] + PATCH_SIZE[0]) / self.valid_mask_scale[0]))
            mined_end_y = int(round((patch.coordinates[1] + PATCH_SIZE[1]) / self.valid_mask_scale[1]))

            self.mined_mask[
            self.min_bound_check(mined_start_x):self.width_bound_check(mined_end_x),
            self.min_bound_check(mined_start_y):self.width_bound_check(mined_end_y)
            ] = True

            self.patches.append(patch)

            if self.label_map is not None:
                lm_patch = self.pull_from_LM(patch)
                self.label_map_patches.append(lm_patch)

            return True

        except Exception as e:
            print(e)
            return False

    def add_next_patch(self):
        """
        Add patch to manager
        :param patch: Patch object to add to set of patches
        :return: None
        TODO: test hashing to ensure same patch won't be added
        """
        if self.valid_mask is None:
            # Find indices on filled mask, then multiply by real scale to get actual coordinates
            x_value = np.random.choice(self.slide_dims[0], 1)
            y_value = np.random.choice(self.slide_dims[1], 1)
            coordinates = np.array([x_value, y_value])
            patch = Patch(self.path, self.slide_object, self, coordinates, 0, PATCH_SIZE, "_patch@({},{})_{}x{}.png")

            return self.add_patch(patch)

        else:

            # Find indices on filled mask, then multiply by real scale to get actual coordinates
            try:
                indices = np.argwhere(self.valid_mask)
                # (X/Y get reversed because openslide and np use reversed height/width indexing)
                x_values = np.round(indices[:, 0] * self.valid_mask_scale[0]).astype(int)
                y_values = np.round(indices[:, 1] * self.valid_mask_scale[1]).astype(int)
                num_indices = len(indices.ravel()) // 2
                print("%i indices left " % num_indices, end="\r")
                if READ_TYPE == 'random':
                    choice = np.random.choice(num_indices, 1)
                elif READ_TYPE == 'sequential':
                    choice = 0
                else:
                    print("Unrecognized read type %s" % READ_TYPE)
                    exit(1)
                coordinates = np.array([x_values[choice], y_values[choice]]).ravel()
                patch = Patch(self.path, self.slide_object, self, coordinates, 0, PATCH_SIZE, "_patch@({},{})_{}x{}.png")
                return self.add_patch(patch)
            except:
                return False

    def remove_patch(self, patch):
        return self.patches.remove(patch)

    def min_bound_check(self, num):
        return max(num, 0)

    def height_bound_check(self, num):
        return min(num, self.slide_dims[0])

    def width_bound_check(self, num):
        return min(num, self.slide_dims[1])

    def add_patch_criteria(self, patch_validity_check):
        self.valid_patch_checks.append(patch_validity_check)

    def mine_patches(self, output_directory, n_patches, output_csv=None, n_jobs=100, save=True, value_map=VALUE_MAP):
        if save:
            os.makedirs(output_directory, exist_ok=True)

        if output_csv is None:
            print("Creating output csv")
            csv_filename = output_directory + self.path[:self.path.rindex(".")].split("/")[-1] + ".csv"
        else:
            csv_filename = output_csv
        output = open(csv_filename, "a")
        if self.label_map is not None:
            output.write("Slide Patch path, Label Map Patch path, Label Map Result\n")
        else:
            output.write("Slide Patch path\n")

        if n_patches == -1:
            n_patches = np.Inf

        _save_patch_partial = partial(_save_patch,
                                      output_directory=output_directory,
                                      save=save,
                                      check_if_valid=True)
        n_completed = 0
        saturated = False

        while n_patches - n_completed > 0 and not saturated:
            # If there is a patch quota given...
            if n_patches != np.Inf:
                # Generate feasible patches until quota is reached or error is raised
                for _ in range(n_patches - n_completed):
                    if not self.add_next_patch():
                        print("\nCould not add new patch, breaking.")
                        break
                else:
                    print("")  # Fixes spacing in case it breaks. Inelegant but I'll fix later

                # If the quota is not met, assume the slide is saturated
                if len(self.patches) != n_patches - n_completed:
                    print("Slide has reached saturation: No more non-overlapping patches to be found.\n"
                          "Change SHOW_MINED in config.py to True to see patch locations.\n"
                          "Alternatively, change READ_TYPE to 'sequential' for greater mining effiency.")
                    saturated = True
            # If there is no patch quota given, add patches until saturation.
            else:
                while True:
                    if not self.add_next_patch():
                        print("\nCould not add new patch, breaking.")
                        break

                if len(self.patches) != n_patches - n_completed:
                    print("Slide has reached saturation: No more non-overlapping patches to be found.\n"
                          "Change SHOW_MINED in config.py to True to see patch locations.\n"
                          "Alternatively, change READ_TYPE to 'sequential' for greater mining effiency.")
                    saturated = True

            os.makedirs(output_directory + self.slide_subfolder, exist_ok=True)
            print("Saving patches:")
            with concurrent.futures.ThreadPoolExecutor(n_jobs) as executor:
                np_slide_futures = list(
                    tqdm(
                        executor.map(_save_patch_partial, self.patches),
                        total=len(self.patches),
                        unit="pchs",
                    )
                )

                self.patches = list()
                np_slide_futures = np.array(np_slide_futures)
                try:
                    successful_indices = np.argwhere(np_slide_futures[:, 0])
                except Exception as e:
                    print(e)
                    print("Setting successful indices to []")
                    successful_indices = []

            # Find all successfully saved patches, copy and extract from label map.
            if self.label_map is not None:
                _lm_save_patch_partial = partial(_save_patch,
                                                 output_directory=output_directory,
                                                 save=save,
                                                 check_if_valid=False,
                                                 patch_processor=get_nonzero_percent,
                                                 value_map=value_map)

                for i in successful_indices:
                    lm_patches = []
                    slide_patch = np_slide_futures[i][0][1]
                    lm_patch = slide_patch.copy()
                    lm_patch._slide_path = self.label_map
                    lm_patches.append(lm_patch)


                print("Saving label maps:")
                os.makedirs(output_directory + self.label_map_subfolder, exist_ok=True)
                with concurrent.futures.ThreadPoolExecutor(100) as executor:
                    lm_futures = list(
                        tqdm(
                            executor.map(_lm_save_patch_partial, self.label_map_patches),
                            total=len(self.label_map_patches),
                            unit="pchs",
                        )
                    )
                np_lm_futures = np.array(lm_futures)

            successful = np.count_nonzero(np_slide_futures == True)
            print("{}/{} valid patches found in this run.".format(successful, n_patches))
            n_completed += successful

            for index in successful_indices:
                if self.label_map is not None:
                    slide_patch_path = np_slide_futures[:, 1][index][0].get_patch_location(output_directory)
                    lm_patch_path = np_lm_futures[:, 1][index][0].get_patch_location(output_directory)
                    lm_result = np_lm_futures[:, 2][index][0]
                    output.write("{},{},{}\n".format(slide_patch_path, lm_patch_path, lm_result))
                else:
                    path_path = np_slide_futures[:, 1][index][0].get_patch_location(output_directory)
                    output.write("{}\n".format(path_path))


        if output_csv is not None:
            output.close()
        print("Done!")

    def save_predefined_patches(self, output_directory, patch_coord_csv, value_map=VALUE_MAP, n_jobs=40):
        # Todo, port to pandas or something more sophisticated?
        with open(patch_coord_csv, "r") as input_csv:
            for line in input_csv:
                try:
                    x, y = [int(val) for val in line.split(",")]
                    os.makedirs(output_directory + self.slide_subfolder, exist_ok=True)
                    _save_patch_partial = partial(_save_patch, output_directory=output_directory,
                                                  save=True,
                                                  check_if_valid=False)

                    patch = Patch(self.path, self.slide_object, self, (x, y), 0, PATCH_SIZE, "_patch@({},{})_{}x{}.png")
                    self.patches.append(patch)

                    if self.label_map is not None:
                        lm_patch = self.pull_from_LM(patch)
                        self.label_map_patches.append(lm_patch)
                except:
                    pass

        os.makedirs(output_directory + self.slide_subfolder, exist_ok=True)
        print("Saving slide patches:")
        with concurrent.futures.ThreadPoolExecutor(n_jobs) as executor:
            futures = list(
                tqdm(
                    executor.map(_save_patch_partial, self.patches),
                    total=len(self.patches),
                    unit="pchs",
                )
            )

        print("Saving label maps:")
        if self.label_map is not None:
            os.makedirs(output_directory + self.label_map_subfolder, exist_ok=True)
            _lm_save_patch_partial = partial(_save_patch,
                                             output_directory=output_directory,
                                             save=True,
                                             check_if_valid=False,
                                             patch_processor=get_nonzero_percent,
                                             value_map=value_map)
            with concurrent.futures.ThreadPoolExecutor(n_jobs) as executor:
                futures = list(
                    tqdm(
                        executor.map(_lm_save_patch_partial, self.label_map_patches),
                        total=len(self.label_map_patches),
                        unit="pchs",
                    )
                )

    def pull_from_LM(self, slide_patch):
        lm_patch = slide_patch.copy()
        lm_patch.set_slide(self.label_map)
        lm_patch.slide_object = self.label_map_object
        lm_patch.output_suffix = "_patch@({},{})_{}x{}_LM.png"

        return lm_patch


def _save_patch(patch, output_directory, save, check_if_valid=True, patch_processor=None, value_map=None):
    return patch.save(out_dir=output_directory,
                      save=save,
                      check_if_valid=check_if_valid,
                      process_method=patch_processor,
                      value_map=value_map)
