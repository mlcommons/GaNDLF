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
from typing import Union

class PatchManager:
    def __init__(self, filename, output_dir):
        """
        Initialization for PatchManager
        @param filename: name of main WSI.
        """
        self.output_dir = output_dir
        self.set_slide_path(filename)
        self.patches = list()
        self.lm_patches = list()
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
        tissue_threshold = config.get("tissue_threshold", 0.5)  # Minimum tissue content threshold

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

            # Create label map patches for valid patches
            if self.label_map is not None:
                for patch in self.patches:
                    lm_patch = self.pull_from_label_map(patch)
                    self.label_map_patches.append(lm_patch)

            # Save patches
            output_dir_slide_folder = os.path.join(self.output_dir, self.slide_folder)
            Path(output_dir_slide_folder).mkdir(parents=True, exist_ok=True)
            
            if self.label_map is not None:
                output_dir_mask_folder = os.path.join(
                    self.output_dir, self.label_map_folder
                )
                Path(output_dir_mask_folder).mkdir(parents=True, exist_ok=True)

            # Define data type for storing results
            dt = np.dtype([
                ('bool_val', bool),
                ('patch_obj', Patch),
                ('string_val', 'U100'),
                ('label_obj', object)
            ])

            def _save_patch_wrapper(patch_and_lm):
                patch, lm_patch = patch_and_lm
                return _save_patch(
                    patch=patch,
                    label_map_patch=lm_patch,
                    output_directory=self.output_dir,
                    save=save,
                    check_if_valid=True,
                    patch_processor=get_patch_class_proportions if self.label_map is not None else None,
                    value_map=value_map,
                )

            print("Saving patches:")

            # Process patches in parallel
            if len(self.label_map_patches) > 0:
                patch_iterable = zip(self.patches, self.label_map_patches)
            else:
                patch_iterable = zip(self.patches, [None] * len(self.patches))

            with concurrent.futures.ThreadPoolExecutor(n_jobs) as executor:
                np_slide_futures = list(
                    tqdm(
                        executor.map(_save_patch_wrapper, patch_iterable),
                        total=len(self.patches),
                        unit="pchs",
                    )
                )

                self.patches = list()
                self.label_map_patches = list()
                
                # Convert to structured numpy array
                np_slide_futures = np.array([(f[0], f[1], f[2], f[3] if len(f) > 3 else None)
                                             for f in np_slide_futures], dtype=dt)
                
                try:
                    successful_indices = np.argwhere(np_slide_futures['bool_val']).ravel()
                except Exception as e:
                    print("Error:", e, "Setting successful indices to []")
                    successful_indices = []
            successful = np.count_nonzero(np_slide_futures['bool_val'])
            print(
                "{}/{} valid patches found in this run.".format(successful, len(np_slide_futures))
            )
            n_completed += successful

            # Generate CSV with patch information
            if len(successful_indices) > 0:
                new_df_rows = []
                for index in successful_indices:
                    new_row = {}
                    if self.save_subjectID:
                        new_row.update({"SubjectID": self.subjectID})

                    slide_patch = np_slide_futures[index]['patch_obj']
                    
                    # Add slide patch information
                    slide_patch_path = slide_patch.get_patch_path(self.output_dir, False)
                    new_row.update({"SlidePatchPath": slide_patch_path})
                    
                    # Add label map information if available
                    if self.label_map is not None and np_slide_futures[index]['label_obj'] is not None:
                        # Create label map patch path based on slide patch
                        lm_patch = self.pull_from_label_map(slide_patch)
                        lm_patch_path = lm_patch.get_patch_path(self.output_dir, False)
                        lm_result = np_slide_futures[index]['string_val']
                        new_row.update({
                            self.mask_header: lm_patch_path,
                            "PatchComposition": lm_result,
                        })

                    # Add coordinates
                    patch_coords = slide_patch.coordinates
                    new_row.update({"PatchCoordinatesX": patch_coords[1]})
                    new_row.update({"PatchCoordinatesY": patch_coords[0]})

                    new_df_rows.append(new_row)

                # Create and save dataframe
                new_df = pd.DataFrame(new_df_rows)
                output_df = pd.concat([output_df, new_df])

        output_df.to_csv(csv_filename, index=False)

        print("Done!")
        return np_slide_futures


    def mine_patch_grid(self, config, output_csv=None):
        # Parse configuration
        patch_size = config["patch_size"]
        overlap = config["overlap_factor"]
        n_jobs = config["num_workers"]
        save_images = config["save_patches"]
        value_map = config["value_map"]
        tissue_threshold = config.get("tissue_threshold", 0.5)  # Minimum tissue content threshold

        # Calculate stride (step size) based on patch size and overlap
        stride_y = int(patch_size[0] * (1 - overlap))
        stride_x = int(patch_size[1] * (1 - overlap))

        # Ensure minimum stride of 1 pixel
        stride_y = max(1, stride_y)
        stride_x = max(1, stride_x)

        # Set default CSV filename if not provided
        csv_filename = os.path.join(self.output_dir, "list.csv")
        if output_csv is not None:
            csv_filename = output_csv

        # Read existing CSV if it exists
        output_df = pd.DataFrame()
        try:
            if os.path.exists(csv_filename) and os.path.isfile(csv_filename):
                output_df = pd.read_csv(csv_filename)
            else:
                output_df = pd.DataFrame()
        except pd.errors.EmptyDataError as e:
            print(e)

        print("Generating tissue-containing patches...")

        # Get slide dimensions
        slide_height, slide_width = self.slide_dims[1], self.slide_dims[0]

        # Use the precomputed tissue mask if available, otherwise generate it
        if self.valid_mask is None:
            print("Warning: No tissue mask provided, processing all patches (inefficient)")
            tissue_mask = np.ones((slide_height // self.valid_mask_scale[1],
                                   slide_width // self.valid_mask_scale[0]), dtype=bool)
            mask_scale = (self.valid_mask_scale[0], self.valid_mask_scale[1])
        else:
            print("Using precomputed tissue mask")
            tissue_mask = self.valid_mask
            mask_scale = self.valid_mask_scale

        # Create a list to store all of our patches
        valid_patch_positions = []

        # Calculate downsampled grid parameters
        ds_patch_height = int(patch_size[0] / mask_scale[1])
        ds_patch_width = int(patch_size[1] / mask_scale[0])
        ds_stride_y = int(stride_y / mask_scale[1])
        ds_stride_x = int(stride_x / mask_scale[0])

        # Calculate number of tiles in downsampled space
        ds_height, ds_width = tissue_mask.shape
        n_tiles_y = (ds_height - ds_patch_height + ds_stride_y) // ds_stride_y
        n_tiles_x = (ds_width - ds_patch_width + ds_stride_x) // ds_stride_x

        print(f"Searching {n_tiles_y} × {n_tiles_x} positions in downsampled space")

        # Identify tissue-containing positions
        total_positions = 0
        tissue_positions = 0

        for y in range(0, ds_height - ds_patch_height + 1, ds_stride_y):
            for x in range(0, ds_width - ds_patch_width + 1, ds_stride_x):
                total_positions += 1

                # Extract the patch from the mask
                mask_patch = tissue_mask[y:y + ds_patch_height, x:x + ds_patch_width]

                # Calculate tissue percentage
                if mask_patch.size > 0:
                    tissue_percentage = np.sum(mask_patch) / mask_patch.size
                else:
                    tissue_percentage = 0

                # If patch meets threshold, add to valid positions
                if tissue_percentage >= tissue_threshold:
                    tissue_positions += 1

                    # Convert coordinates back to original resolution
                    orig_y = int(y * mask_scale[1])
                    orig_x = int(x * mask_scale[0])

                    valid_patch_positions.append((orig_y, orig_x))

        print(f"Found {tissue_positions}/{total_positions} positions with sufficient tissue content")

        # Create Patch objects for valid positions
        self.patches = []
        for y, x in valid_patch_positions:
            # Create a patch at this position
            patch = Patch(
                slide_path=self.img_path,
                slide_object=self.slide_object,
                manager=self,
                coordinates=np.array([y, x]),
                level=0,
                size=patch_size,
                output_suffix="_patch_{}-{}.png",
            )

            self.patches.append(patch)

            if self.label_map is not None:
                self.label_map_patches.append(self.pull_from_label_map(patch))

        print(f"Created {len(self.patches)} patch objects. Validating and saving patches...")

        # Create output directories
        output_dir_slide_folder = os.path.join(self.output_dir, self.slide_folder)
        Path(output_dir_slide_folder).mkdir(parents=True, exist_ok=True)

        # Define data type for storing results
        dt = np.dtype([
            ('bool_val', bool),
            ('patch_obj', Patch),
            ('string_val', 'U100'),
            ('label_obj', object)# Unicode string
        ])


        # Prepare the save function with partial
        if save_images:
            _save_patch_partial = partial(
                _save_patch,
                output_directory=self.output_dir,
                check_if_valid=True,
            )
        else:
            _save_patch_partial = partial(
                _get_patch,
                value_map=value_map
            )

        print(f"Processing and saving {len(self.patches)} patches:")

        # Process patches in parallel
        if len(self.label_map_patches) > 0:
            patch_iterable = zip(self.patches, self.label_map_patches)
        else:
            patch_iterable = zip(self.patches, [None] * len(self.patches))

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            np_slide_futures = list(
                tqdm(
                    executor.map(lambda args: _save_patch_partial(*args),
                                 patch_iterable),
                    total=len(self.patches),
                    unit="pchs",
                )
            )

            self.patches = []  # Clear patches to free memory

            # Convert to structured numpy array
            np_slide_futures = np.array([(f[0], f[1], f[2], f[3] if len(f) > 2 else "")
                                         for f in np_slide_futures], dtype=dt)

            try:
                successful_indices = np.argwhere(np_slide_futures['bool_val']).ravel()
            except Exception as e:
                print("Error:", e, "Setting successful indices to []")
                successful_indices = []

        # Generate CSV with patch information
        if len(successful_indices) > 0:
            new_df_rows = []
            for index in successful_indices:
                new_row = {}
                if self.save_subjectID:
                    new_row.update({"SubjectID": self.subjectID})

                slide_patch = np_slide_futures[index]['patch_obj']

                # Add slide patch information
                slide_patch_path = slide_patch.get_patch_path(self.output_dir, False)
                new_row.update({"SlidePatchPath": slide_patch_path})

                # Add coordinates
                patch_coords = slide_patch.coordinates
                new_row.update({"PatchCoordinatesX": patch_coords[1]})
                new_row.update({"PatchCoordinatesY": patch_coords[0]})

                new_df_rows.append(new_row)

            # Create and save dataframe
            new_df = pd.DataFrame(new_df_rows)
            output_df = pd.concat([output_df, new_df])
            output_df.to_csv(csv_filename, index=False)

        # Report statistics
        successful = np.count_nonzero(np_slide_futures['bool_val'])
        print(f"{successful}/{len(np_slide_futures)} valid patches found in this run.")

        return np_slide_futures


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
    patch: Patch,
    label_map_patch: Patch = None,
    output_directory=None,
    save=True,
    check_if_valid=True,
    patch_processor=None,
    value_map=None,
):
    if save:
        return patch.save(
            out_dir=output_directory,
            check_if_valid=check_if_valid,
            process_method=patch_processor,
            value_map=value_map,
        )
    else:
        return patch.validate(
            label_map=label_map_patch,
            process_method=patch_processor,
            value_map=value_map,
            check_validity=check_if_valid
        )

def _get_patch(
    patch: Patch,
    label_map_patch: Patch,
    value_map=None,
    process_method=None
):
    return patch.validate(label_map=label_map_patch, process_method=process_method, value_map=value_map)
