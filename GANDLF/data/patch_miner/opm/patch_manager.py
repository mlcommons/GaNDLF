import concurrent.futures
import os
from functools import partial
from .patch import Patch
from .utils import get_patch_class_proportions, convert_to_tiff
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import openslide


class PatchManager:
    def __init__(self, filename, output_dir):
        """
        Initialization for PatchManager
        @param filename: name of main WSI.
        """
        self.output_dir = output_dir
        self.set_slide_path(filename)
        self.patches = list()
        self.slide_folder = Path(filename).stem
        self.valid_mask = None
        self.mined_mask = None
        self.valid_mask_scale = (0, 0)
        self.valid_patch_checks = []
        self.label_map = None
        self.label_map_object = None
        self.label_map_folder = None
        self.label_map_patches = list()
        self.subjectID = None
        self.save_subjectID = False
        self.image_header = "SlidePatchPath"
        self.mask_header = "LabelMapPatchPath"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def set_subjectID(self, subjectID):
        self.subjectID = str(subjectID)
        self.save_subjectID = True

    def set_slide_path(self, filename):
        self.img_path = filename
        self.img_path = convert_to_tiff(self.img_path, self.output_dir, "img")
        self.slide_object = openslide.open_slide(self.img_path)
        self.slide_dims = self.slide_object.dimensions

    def set_label_map(self, path):
        """
        Add associated label map to Patch Manager.
        @param path: path to label map.
        """
        self.label_map = convert_to_tiff(path, self.output_dir, "mask")
        self.label_map_object = openslide.open_slide(self.label_map)

        assert all(
            x == y for x, y in zip(self.label_map_object.dimensions, self.slide_dims)
        ), "Label map must have same dimensions as main slide."
        self.label_map_folder = Path(path).stem

    def set_valid_mask(self, mask, scale=(1, 1)):
        self.valid_mask = mask
        self.mined_mask = np.zeros_like(mask)
        self.valid_mask_scale = scale

    def add_patch(self, patch, overlap_factor, patch_size):
        """
        Add patch to manager and take care of self.mined_mask update so it doesn't pull the same patch twice.

        This method works by first finding the coordinates of the upper-left corner of the patch. It then multiples the
        inverse overlap factor (0 = full overlap -> 1 = no overlap) by the patch dimensions to find the region that
        should be excluded from being called again. It does this from the top-left to bottom-right of the coordinate to
        include all space that would cause a patch overlap.

        TODO: Rework the math so the (x,y) coordinate is the center of the patch-- not the top left.
        :param patch: Patch object to add to set of patches
        :return: None
        """
        try:
            # Set inverse overlap factor
            inverse_overlap_factor = 1 - overlap_factor

            # Find the coordinates on the valid mask, make sure to scale
            valid_start_x = int(
                round(
                    (
                        patch.coordinates[0]
                        - int(round((patch_size[0] + 1) * inverse_overlap_factor))
                    )
                    / self.valid_mask_scale[0]
                )
            )
            valid_start_y = int(
                round(
                    (
                        patch.coordinates[1]
                        - int(round((patch_size[1] + 1) * inverse_overlap_factor))
                    )
                    / self.valid_mask_scale[1]
                )
            )

            # If the user specifies anything other than 100% overlap being okay, update the valid mask to remove an
            # already called region.
            if overlap_factor != 1:
                # (bounds checking)
                valid_end_x = int(
                    round(
                        (
                            patch.coordinates[0]
                            + int(round(patch_size[0] * inverse_overlap_factor))
                        )
                        / self.valid_mask_scale[0]
                    )
                )
                valid_end_y = int(
                    round(
                        (
                            patch.coordinates[1]
                            + int(round(patch_size[1] * inverse_overlap_factor))
                        )
                        / self.valid_mask_scale[1]
                    )
                )
                # Set the valid mask values to False so a coordinate that would cause overlap cannot be called later.
                self.valid_mask[
                    max(valid_start_x, 0) : self.width_bound_check(valid_end_x),
                    max(valid_start_y, 0) : self.height_bound_check(valid_end_y),
                ] = False
            else:
                # If the user is okay with 100% overlap, just remove the single pixel of the coordinate and change only the starting index
                self.valid_mask[valid_start_x, valid_start_y] = False

            mined_start_x = int(
                round((patch.coordinates[0]) / self.valid_mask_scale[0])
            )
            mined_start_y = int(
                round((patch.coordinates[1]) / self.valid_mask_scale[1])
            )
            mined_end_x = int(
                round((patch.coordinates[0] + patch_size[0]) / self.valid_mask_scale[0])
            )
            mined_end_y = int(
                round((patch.coordinates[1] + patch_size[1]) / self.valid_mask_scale[1])
            )

            # Update the mined mask
            self.mined_mask[
                max(mined_start_x, 0) : self.width_bound_check(mined_end_x),
                max(mined_start_y, 0) : self.width_bound_check(mined_end_y),
            ] = True
            # Append this patch to the list of patches to be saved
            self.patches.append(patch)
            return True

        except Exception as e:
            # If it fails for any reason, print the exception and return
            print("Exception thrown when adding patch:", e)
            return False

    def find_next_patch(self, patch_size, read_type, overlap_factor):
        """
        Select the next patch location.
        @return: True if patch is successfully saved, False if not.
        """
        # If there is no valid mask, select anywhere on the slide.
        if self.valid_mask is None:
            # Find indices on filled mask, then multiply by real scale to get actual coordinates
            x_value = np.random.choice(self.slide_dims[0], 1)
            y_value = np.random.choice(self.slide_dims[1], 1)
            coordinates = np.array([x_value, y_value])
            patch = Patch(
                self.img_path,
                self.slide_object,
                self,
                coordinates,
                0,
                patch_size,
                "_patch_{}-{}.png",
            )

            return self.add_patch(patch, overlap_factor, patch_size)

        else:
            # Find indices on filled mask, then multiply by real scale to get actual coordinates
            try:
                indices = np.argwhere(self.valid_mask)
                # (X/Y get reversed because OpenSlide and np use reversed height/width indexing)
                x_values = np.round(indices[:, 0] * self.valid_mask_scale[0]).astype(
                    int
                )
                y_values = np.round(indices[:, 1] * self.valid_mask_scale[1]).astype(
                    int
                )
                num_indices = len(indices.ravel()) // 2
                print("%i indices left " % num_indices, end="\r")
                # Find index of coordinates to select for patch

                assert read_type in ["random", "sequential"], (
                    "Unrecognized read type %s" % read_type
                )
                if read_type == "random":
                    choice = np.random.choice(num_indices, 1)
                elif read_type == "sequential":
                    choice = 0

                coordinates = np.array([x_values[choice], y_values[choice]]).ravel()

                patch = Patch(
                    slide_path=self.img_path,
                    slide_object=self.slide_object,
                    manager=self,
                    coordinates=coordinates,
                    level=0,
                    size=patch_size,
                    output_suffix="_patch_{}-{}.png",
                )
                return self.add_patch(patch, overlap_factor, patch_size)
            except Exception as e:
                print("Exception thrown when adding next patch:", e)
                return False

    def remove_patch(self, patch):
        return self.patches.remove(patch)

    def height_bound_check(self, num):
        return min(num, self.slide_dims[0])

    def width_bound_check(self, num):
        return min(num, self.slide_dims[1])

    def add_patch_criteria(self, patch_validity_check):
        """
        Add check for if the patch can be saved.
        @param patch_validity_check: A function that takes only an image as an argument and returns True if the patch
            passes the check, False if the patch should be rejected.
        """
        self.valid_patch_checks.append(patch_validity_check)

    def set_image_header(self, image_header):
        self.image_header = image_header

    def set_mask_header(self, mask_header):
        self.mask_header = mask_header

    def _is_patch_extraction_done(self, n_patches, n_completed):
        # If the quota is not met, assume the slide is saturated
        slide_saturation_message = "Slide has reached saturation, and no more non-overlapping patches to be found; try 'read_type: sequential' in config for greater mining efficiency."
        if len(self.patches) != n_patches - n_completed:
            print(slide_saturation_message)
            return True
        return False

    def mine_patches(self, config, output_csv=None):
        """
        Main loop of the program. This generates patch locations and attempts to save them until the slide is either
        saturated or the quota has been met.

        This is essentially a large loop that takes the following steps:
        [LOOP START]
            1) Find potential patch coordinates from self.valid_mask
                - Adds each patch to self.patches
            2) Read and save all patches stored in self.patches
                - If patch CANNOT be saved, return [False, Patch, ""]
                    > this is due to either being rejected by methods added by add_patch_criteria or an error.
                - If patch WAS saved, return [False, Patch, patch_processor(patch)]
            IF label_map is provided:
                3) Pull successfully saved slide patches from corresponding label map locations.
                4) Save all pulled label map patches
                    - this does NOT check patches for validity
            IF slide is saturated ==> EXIT LOOP
            IF quota is met       ==> EXIT LOOP
        [REPEAT LOOP]

        @param n_patches: either an int for the number of patches, or -1 for mining until exhaustion.
        @param output_csv: The path of the output .csv to write. If none specified, put it in the output folder.
        @param n_jobs: Number of threads to launch.
        @param save: 'Dummy' run of patch extraction if False.
        @param value_map: Dictionary for value swapping.
        """

        # initialize defaults
        n_patches = config["num_patches"]
        n_jobs = config["num_workers"]
        save = config["save_patches"]
        value_map = config["value_map"]

        csv_filename = os.path.join(self.output_dir, "list.csv")

        if output_csv is not None:
            csv_filename = output_csv

        output_df = pd.DataFrame()
        try:
            if os.path.exists(csv_filename) and os.path.isfile(csv_filename):
                output_df = pd.read_csv(csv_filename)
            else:
                output_df = pd.DataFrame()
        except pd.errors.EmptyDataError as e:
            print(e)

        n_patches = np.Inf if n_patches == -1 else n_patches

        n_completed = 0
        saturated = False

        while n_patches - n_completed > 0 and not saturated:
            # If there is a patch quota given...
            if n_patches != np.Inf:
                # Generate feasible patches until quota is reached or error is raised
                for _ in range(n_patches - n_completed):
                    if not self.find_next_patch(
                        patch_size=config["patch_size"],
                        read_type=config["read_type"],
                        overlap_factor=config["overlap_factor"],
                    ):
                        print("\nCould not add new patch, breaking.")
                        break

                # If the quota is not met, assume the slide is saturated
                saturated = self._is_patch_extraction_done(n_patches, n_completed)
            # If there is no patch quota given, add patches until saturation.
            else:
                while True:
                    if not self.find_next_patch(
                        patch_size=config["patch_size"],
                        read_type=config["read_type"],
                        overlap_factor=config["overlap_factor"],
                    ):
                        print("\nCould not add new patch, breaking.")
                        break

                saturated = self._is_patch_extraction_done(n_patches, n_completed)

            # Save patches
            output_dir_slide_folder = os.path.join(self.output_dir, self.slide_folder)
            Path(output_dir_slide_folder).mkdir(parents=True, exist_ok=True)
            _save_patch_partial = partial(
                _save_patch,
                output_directory=self.output_dir,
                save=save,
                check_if_valid=True,
            )

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
                    successful_indices = np.argwhere(np_slide_futures[:, 0]).ravel()
                except Exception as e:
                    print("Error:", e, "Setting successful indices to []")
                    successful_indices = []

            # Find all successfully saved patches, copy and extract from label map.
            if self.label_map is not None:
                for i in successful_indices:
                    slide_patch = np_slide_futures[i, 1]
                    lm_patch = self.pull_from_label_map(slide_patch)
                    self.label_map_patches.append(lm_patch)

                print("Saving label maps:")
                output_dir_mask_folder = os.path.join(
                    self.output_dir, self.label_map_folder
                )
                Path(output_dir_mask_folder).mkdir(parents=True, exist_ok=True)

                _lm_save_patch_partial = partial(
                    _save_patch,
                    output_directory=self.output_dir,
                    save=save,
                    check_if_valid=False,
                    patch_processor=get_patch_class_proportions,
                    value_map=value_map,
                )
                with concurrent.futures.ThreadPoolExecutor(
                    config["num_workers"]
                ) as executor:
                    lm_futures = list(
                        tqdm(
                            executor.map(
                                _lm_save_patch_partial, self.label_map_patches
                            ),
                            total=len(self.label_map_patches),
                            unit="pchs",
                        )
                    )
                np_lm_futures = np.array(lm_futures)
            successful = np.count_nonzero(np_slide_futures[:, 0])
            print(
                "{}/{} valid patches found in this run.".format(successful, n_patches)
            )
            n_completed += successful

            new_df_rows = []
            for index in successful_indices:
                new_row = {}
                if self.save_subjectID:
                    new_row.update({"SubjectID": self.subjectID})
                if self.label_map is not None:
                    slide_patch_path = np_slide_futures[index, 1].get_patch_path(
                        self.output_dir, False
                    )
                    lm_patch_path = np_lm_futures[index, 1].get_patch_path(
                        self.output_dir, False
                    )
                    lm_result = np_lm_futures[index, 2]
                    new_row.update(
                        {
                            self.image_header: slide_patch_path,
                            self.mask_header: lm_patch_path,
                            "PatchComposition": lm_result,
                        }
                    )

                slide_patch_path = np_slide_futures[index, 1].get_patch_path(
                    self.output_dir, False
                )
                new_row.update({"SlidePatchPath": slide_patch_path})

                patch_coords = np_slide_futures[index, 1].coordinates
                new_row.update({"PatchCoordinatesX": patch_coords[1]})
                new_row.update({"PatchCoordinatesY": patch_coords[0]})

                new_df_rows.append(new_row)

            new_df = pd.DataFrame(new_df_rows)
            # Concatenate in case there is a pre-existing dataframe
            output_df = pd.concat([output_df, new_df])

        output_df.to_csv(csv_filename, index=False)

        print("Done!")

    ### commenting out functionality for now, will be ported to a separate script
    # def save_predefined_patches(
    #     self,
    #     patch_coord_csv,
    #     config,
    #     x_coord_col="PatchCoordinatesX",
    #     y_coord_col="PatchCoordinatesY",
    # ):
    #     """

    #     @param output_directory:
    #     @param patch_coord_csv:
    #     @param value_map:
    #     @param n_jobs:
    #     @return:
    #     """

    #     value_map = config["value_map"]
    #     patch_size = config["patch_size"]
    #     n_jobs = config["num_workers"]

    #     output_dir_slide_folder = os.path.join(self.output_dir, self.slide_folder)
    #     Path(output_dir_slide_folder).mkdir(parents=True, exist_ok=True)
    #     # Todo, port to pandas or something more sophisticated?
    #     input_df = pd.read_csv(patch_coord_csv)
    #     for idx, row in input_df.iterrows():
    #         x, y = row[x_coord_col], row[y_coord_col]
    #         patch = Patch(
    #             self.img_path,
    #             self.slide_object,
    #             self,
    #             [y, x],
    #             0,
    #             patch_size,
    #             "_patch_{}-{}.png",
    #         )
    #         self.patches.append(patch)

    #         if self.label_map is not None:
    #             lm_patch = self.pull_from_label_map(patch)
    #             self.label_map_patches.append(lm_patch)

    #     _save_patch_partial = partial(
    #         _save_patch,
    #         output_directory=output_dir_slide_folder,
    #         save=True,
    #         check_if_valid=False,
    #     )

    #     print("Saving slide patches:")
    #     with concurrent.futures.ThreadPoolExecutor(n_jobs) as executor:
    #         list(
    #             tqdm(
    #                 executor.map(_save_patch_partial, self.patches),
    #                 total=len(self.patches),
    #                 unit="pchs",
    #             )
    #         )

    #     if self.label_map is not None:
    #         print("Saving label maps:")
    #         _lm_save_patch_partial = partial(
    #             _save_patch,
    #             output_directory=output_dir_slide_folder,
    #             save=True,
    #             check_if_valid=False,
    #             patch_processor=get_patch_class_proportions,
    #             value_map=value_map,
    #         )
    #         with concurrent.futures.ThreadPoolExecutor(n_jobs) as executor:
    #             list(
    #                 tqdm(
    #                     executor.map(_lm_save_patch_partial, self.label_map_patches),
    #                     total=len(self.label_map_patches),
    #                     unit="pchs",
    #                 )
    #             )

    def pull_from_label_map(self, slide_patch):
        """
        Copy a patch from the slide and use the coordinates to pull a corresponding patch from the LM.
        @param slide_patch:
        @return:
        """
        lm_patch = slide_patch.copy()
        lm_patch.set_slide(self.label_map)
        lm_patch.slide_object = self.label_map_object
        lm_patch.output_suffix = "_patch_{}-{}_LM.png"

        return lm_patch


def _save_patch(
    patch,
    output_directory,
    save,
    check_if_valid=True,
    patch_processor=None,
    value_map=None,
):
    return patch.save(
        out_dir=output_directory,
        save=save,
        check_if_valid=check_if_valid,
        process_method=patch_processor,
        value_map=value_map,
    )
