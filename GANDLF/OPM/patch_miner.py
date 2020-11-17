import time
import argparse
from opm.patch_manager import *
from opm.utils import *
import numpy as np
from PIL import Image
from functools import partial

Image.MAX_IMAGE_PIXELS = None
from pathlib import Path
import warnings

warnings.simplefilter("ignore")


def generate_initial_mask(slide_path):
    """
    Helper method to generate random coordinates within a slide
    :param slide_path: Path to slide (str)
    :param num_patches: Number of patches you want to generate
    :return: list of n (x,y) coordinates
    """
    # Open slide and get properties
    slide = openslide.open_slide(slide_path)
    slide_dims = slide.dimensions

    # Call thumbnail for effiency, calculate scale relative to whole slide
    slide_thumbnail = np.asarray(slide.get_thumbnail((slide_dims[0] // SCALE, slide_dims[1] // SCALE)))
    real_scale = (slide_dims[0] / slide_thumbnail.shape[1], slide_dims[1] / slide_thumbnail.shape[0])

    return tissue_mask(slide_thumbnail), real_scale


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        dest='input_path',
                        help="input path for the tissue",
                        required=True)
    parser.add_argument('-n', '--num_patches',
                        type=int,
                        default=1000,
                        dest='num_patches',
                        help="Number of patches to mine. Set to -1 to mine until saturation. ",
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
    parser.add_argument('-t', '--threads',
                        dest='threads',
                        help="number of threads, by default will use all")

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
        if not out_dir.endswith("/"):
            out_dir += "/"
    # Path to openslide supported file (.svs, .tiff, etc.)
    slide_path = os.path.abspath(args.input_path)

    if not os.path.exists(slide_path):
        raise ValueError("Could not find the slide, could you recheck the path?")

    # Create new instance of slide manager
    manager = PatchManager(slide_path)

    if args.input_csv is None:
        # Generate an initial validity mask
        mask, scale = generate_initial_mask(args.input_path)
        manager.set_valid_mask(mask, scale)
        manager.set_label_map(args.label_map_path)

        # Reject patch if any pixels are transparent
        manager.add_patch_criteria(alpha_channel_check)
        # Reject patch if image dimensions are not equal to PATCH_SIZE
        patch_dims_check = partial(patch_size_check, patch_height=PATCH_SIZE[0], patch_width=PATCH_SIZE[1])
        manager.add_patch_criteria(patch_dims_check)
        # Save patches releases saves all patches stored in manager, dumps to specified output file
        manager.mine_patches(out_dir, n_patches=args.num_patches, output_csv=args.output_csv, n_jobs=NUM_WORKERS,
                             save=do_save_patches)
        print("Total time: {}".format(time.time() - start))
    else:
        manager.save_predefined_patches(out_dir, patch_coord_csv=args.input_csv)

