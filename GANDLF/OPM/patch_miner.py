import time
import argparse
import numpy as np
import warnings
import yaml
import os

import tiffslide
from PIL import Image
from pathlib import Path
from functools import partial
from opm.patch_manager import PatchManager
from opm.utils import alpha_channel_check, patch_size_check, parse_config, generate_initial_mask, get_patch_size_in_microns

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore")


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        dest='input_path',
                        help="input path for the tissue",
                        required=True)
    parser.add_argument('-c', '--config',
                        type=str,
                        dest='config',
                        help="config.yml for running OPM. ",
                        required=True)
    parser.add_argument('-lm', '--label_map_path',
                        dest='label_map_path',
                        help="input path for the label mask")
    parser.add_argument('-o', '--output_path',
                        dest='output_path', default=None,
                        help="output path for the patches")
    parser.add_argument('-ocsv', '--output_csv',
                        dest='output_csv', default=None,
                        help="output path for the csv.")
    parser.add_argument('-icsv', '--input_csv',
                        dest='input_csv', default=None,
                        help="CSV with x,y coordinates of patches to mine.")

    args = parser.parse_args()
    if args.output_path is None:
        do_save_patches = False
        out_dir = ""
    else:
        if not os.path.exists(args.output_path):
            print("Output Directory does not exist, we are creating one for you.")
            Path(args.output_path).mkdir(parents=True, exist_ok=True)

        do_save_patches = True
        out_dir = os.path.abspath(args.output_path)
    # Path to openslide supported file (.svs, .tiff, etc.)
    slide_path = os.path.abspath(args.input_path)

    if not os.path.exists(slide_path):
        raise ValueError("Could not find the slide, could you recheck the path?")

    # Create new instance of slide manager
    manager = PatchManager(slide_path, args.output_path)
    cfg = parse_config(args.config)

    if args.input_csv is None:
        # Generate an initial validity mask
        mask, scale = generate_initial_mask(slide_path, cfg['scale'])
        manager.set_valid_mask(mask, scale)
        if args.label_map_path is not None:
            manager.set_label_map(args.label_map_path)
        
        ## trying to handle mpp
        cfg['patch_size'] = get_patch_size_in_microns(slide_path, cfg['patch_size'], True)

        # Reject patch if any pixels are transparent
        manager.add_patch_criteria(alpha_channel_check)
        # Reject patch if image dimensions are not equal to PATCH_SIZE
        patch_dims_check = partial(patch_size_check, patch_height=cfg['patch_size'][0], patch_width=cfg['patch_size'][1])
        manager.add_patch_criteria(patch_dims_check)
        # Save patches releases saves all patches stored in manager, dumps to specified output file
        manager.mine_patches(output_csv=args.output_csv, config=cfg)
        print("Total time: {}".format(time.time() - start))
    else:
        if args.label_map_path is not None:
            manager.set_label_map(args.label_map_path)
        manager.save_predefined_patches(patch_coord_csv=args.input_csv, config=cfg)

