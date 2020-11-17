import time
import warnings
import argparse
import pandas as pd
import openslide
import numpy as np

from PIL import Image
from GANDLF.OPM.opm.patch_manager import *
from GANDLF.OPM.opm.utils import *
from functools import partial
from pathlib import Path

Image.MAX_IMAGE_PIXELS = None
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
    slide_thumbnail = np.asarray(slide.get_thumbnail((slide_dims[0]//SCALE, slide_dims[1]//SCALE)))
    real_scale = (slide_dims[0]/slide_thumbnail.shape[1], slide_dims[1]/slide_thumbnail.shape[0])


    return tissue_mask(slide_thumbnail), real_scale

def parse_gandlf_csv(fpath):
    df = pd.read_csv(fpath)
    for index, row in df.iterrows():
        yield row['Channel_0'], row['Label']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        dest='input_path',
                        help="input path for the tissue",
                        required=True)
    parser.add_argument('-o', '--output_path',
                        dest='output_path',
                        default=None,
                        required=True,
                        help="output path for the patches")
    args = parser.parse_args()


    if not os.path.exists(args.output_path):
        print("Output Directory does not exist, we are creating one for you.")
        Path(args.output_path).mkdir(parents=True, exist_ok=True)

    out_dir = os.path.abspath(args.output_path)
    if not out_dir.endswith("/"):
        out_dir += "/"

    out_csv_path = out_dir + out_dir.split("/")[-2] + ".csv"

    for slide, label in parse_gandlf_csv(args.input_path):
        start = time.time()

        # Create new instance of slide manager
        manager = PatchManager(slide)
        manager.set_label_map(label)

        # Generate an initial validity mask
        mask, scale = generate_initial_mask(slide)
        print("Setting valid mask...")
        manager.set_valid_mask(mask, scale)
        # Reject patch if any pixels are transparent
        manager.add_patch_criteria(alpha_channel_check)
        # Reject patch if image dimensions are not equal to PATCH_SIZE
        patch_dims_check = partial(patch_size_check, patch_height=PATCH_SIZE[0], patch_width=PATCH_SIZE[1])
        manager.add_patch_criteria(patch_dims_check)
        # Save patches releases saves all patches stored in manager, dumps to specified output file
        manager.mine_patches(out_dir, output_csv=out_csv_path, n_patches=NUM_PATCHES, n_jobs=NUM_WORKERS, save=True)
        print("Total time: {}".format(time.time() - start))

