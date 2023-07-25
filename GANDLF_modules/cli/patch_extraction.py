import os, warnings
from functools import partial
from pathlib import Path

from PIL import Image

from GANDLF.data.patch_miner.opm.patch_manager import PatchManager
from GANDLF.data.patch_miner.opm.utils import (
    alpha_rgb_2d_channel_check,
    patch_size_check,
    parse_config,
    generate_initial_mask,
    get_patch_size_in_microns,
    patch_artifact_check,
    # pen_marking_check,
)
from GANDLF.utils import (
    parseTrainingCSV,
)


def parse_gandlf_csv(fpath):
    df, _ = parseTrainingCSV(fpath, train=False)
    df = df.drop_duplicates()
    # nans can be easily removed using df.dropna(axis=1, how='all')
    # we want to keep them because we want the user to check the CSV instead
    # there might be cases where labels are accidentally removed for some subjects, but not all
    assert (
        df.isnull().values.any() == False
    ), "Data CSV contains null/nan values, please check."
    for _, row in df.iterrows():
        if "Label" in row:
            yield row["SubjectID"], row["Channel_0"], row["Label"]
        else:
            yield row["SubjectID"], row["Channel_0"], None


def patch_extraction(input_path, output_path, config=None):
    """
    This function extracts patches from WSIs.

    Args:
        input_path (str): The input CSV.
        config (Union[str, dict, none]): The input yaml config.
        output_path (_type_): _description_
    """

    Image.MAX_IMAGE_PIXELS = None
    warnings.simplefilter("ignore")

    # initialize default config
    cfg = {}
    if config is not None:
        cfg = config
        if isinstance(config, str):
            cfg = parse_config(config)
    cfg["scale"] = cfg.get("scale", 16)
    cfg["patch_size"] = cfg.get("patch_size", (256, 256))
    original_patch_size = cfg["patch_size"]

    if not os.path.exists(output_path):
        Path(output_path).mkdir(parents=True, exist_ok=True)

    output_path = os.path.abspath(output_path)

    out_csv_path = os.path.join(output_path, "opm_train.csv")

    for sid, slide, label in parse_gandlf_csv(input_path):
        # Create new instance of slide manager
        manager = PatchManager(slide, os.path.join(output_path, str(sid)))
        if label is not None:
            manager.set_label_map(label)
        manager.set_subjectID(str(sid))
        manager.set_image_header("Channel_0")
        manager.set_mask_header("Label")

        cfg["patch_size"] = get_patch_size_in_microns(slide, original_patch_size)

        # Generate an initial validity mask
        mask, scale = generate_initial_mask(slide, cfg["scale"])
        print("Setting valid mask...")
        manager.set_valid_mask(mask, scale)
        # Reject patch if any pixels are transparent
        manager.add_patch_criteria(alpha_rgb_2d_channel_check)
        #manager.add_patch_criteria(pen_marking_check) ### will be added to main code after rigourous experimentation
        manager.add_patch_criteria(patch_artifact_check)
        # Reject patch if image dimensions are not equal to PATCH_SIZE
        patch_dims_check = partial(
            patch_size_check,
            patch_height=cfg["patch_size"][0],
            patch_width=cfg["patch_size"][1],
        )
        manager.add_patch_criteria(patch_dims_check)
        # Save patches releases saves all patches stored in manager, dumps to specified output file
        manager.mine_patches(output_csv=out_csv_path, config=cfg)
