
import time, os, warnings
from functools import partial
from pathlib import Path

import pandas as pd
from PIL import Image

from GANDLF.OPM.opm.patch_manager import PatchManager
from GANDLF.OPM.opm.utils import (
    alpha_channel_check,
    patch_size_check,
    parse_config,
    generate_initial_mask,
)

def parse_gandlf_csv(fpath):
    df = pd.read_csv(fpath, dtype=str)
    df = df.drop_duplicates()
    for _, row in df.iterrows():
        if "Label" in row:
            yield row["SubjectID"], row["Channel_0"], row["Label"]
        else:
            yield row["SubjectID"], row["Channel_0"]


def patch_extraction(args,):
    
    Image.MAX_IMAGE_PIXELS = None
    warnings.simplefilter("ignore")
    
    if args.config is not None:
        cfg = parse_config(args.config)
    else:
        cfg = {}
        cfg["scale"] = 16
        cfg["patch_size"] = (256, 256)

    if not os.path.exists(args.output_path):
        Path(args.output_path).mkdir(parents=True, exist_ok=True)

    args.output_path = os.path.abspath(args.output_path)

    out_csv_path = os.path.join(args.output_path, "opm_train.csv")

    for sid, slide, label in parse_gandlf_csv(args.input_path):
        start = time.time()

        # Create new instance of slide manager
        manager = PatchManager(slide, args.output_path)
        manager.set_label_map(label)
        manager.set_subjectID(sid)
        manager.set_image_header("Channel_0")
        manager.set_mask_header("Label")

        # Generate an initial validity mask
        mask, scale = generate_initial_mask(slide, cfg["scale"])
        print("Setting valid mask...")
        manager.set_valid_mask(mask, scale)
        # Reject patch if any pixels are transparent
        manager.add_patch_criteria(alpha_channel_check)
        # Reject patch if image dimensions are not equal to PATCH_SIZE
        patch_dims_check = partial(
            patch_size_check,
            patch_height=cfg["patch_size"][0],
            patch_width=cfg["patch_size"][1],
        )
        manager.add_patch_criteria(patch_dims_check)
        # Save patches releases saves all patches stored in manager, dumps to specified output file
        manager.mine_patches(output_csv=out_csv_path, config=cfg)