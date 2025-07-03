import argparse
import json
import os

import cv2
import numpy as np
import openslide
import shapely.geometry
from PIL import Image
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
import pandas as pd

def load_annotations(json_path):
    """Load JSON annotations from a file.

    Args:
        json_path: Path to the JSON annotations file.

    """
    with open(json_path, "r") as file:
        return json.load(file)


def extract_polygons(annotations):
    """Extract polygons and their names from annotations.
    Args:
        annotations: List of annotations or a single annotation dictionary.
    """
    if isinstance(annotations, dict):
        annotations = [annotations]  # Wrap dictionary in a list

    polygons = []
    for annotation in annotations:
        for i, element in enumerate(
            annotation.get("annotation", {}).get("elements", [])
        ):
            shape_type = element.get("type", "").lower()
            if shape_type in {"polyline", "polygon"} and element.get("closed", True):
                points = element.get("points", [])
                if points:
                    polygon = shapely.geometry.Polygon([tuple(p[:2]) for p in points])

                    # Label the polygon based on the element label
                    label = element.get("label", {}).get("value")

                    polygons.append((f"{label}_{i}", polygon))
            elif shape_type in {"rectangle", "ellipse"}:
                center = tuple(element.get("center", [0, 0])[:2])
                width = element.get("width", 0)
                height = element.get("height", 0)
                if shape_type == "rectangle":
                    # Calculate rectangle corners based on center, width, and height
                    x0 = center[0] - width / 2
                    x1 = center[0] + width / 2
                    y0 = center[1] - height / 2
                    y1 = center[1] + height / 2
                    points = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
                    polygon = shapely.geometry.Polygon(points)
                elif shape_type == "ellipse":
                    # Approximate ellipse as a polygon
                    radius_x = width / 2
                    radius_y = height / 2
                    ellipse_point = shapely.geometry.Point(center)
                    polygon = ellipse_point.buffer(
                        1, resolution=16
                    )  # Creating a circle
                    polygon = shapely.affinity.scale(
                        polygon, radius_x, radius_y
                    )  # Scaling to ellipse
                label = element.get("label", {}).get("value", "unnamed")
                polygons.append((f"{label}_{i}", polygon))
    return polygons


def patch_check(
    img,
    patch_size=(512, 512),
    intensity_thresh=250,
    intensity_thresh_saturation=5,
    intensity_thresh_b=128,
):
    """
    This function is used to curate patches from the input image. It is used to remove patches that are mostly background or have pen markings.

    Args:
        img (np.ndarray): Input Patch Array to check for artifacts and pen markings.
        patch_size (tuple, optional): Tiling Size of the patch. Defaults to (256, 256).
        intensity_thresh (int, optional): Threshold to check whiteness in the patch. Defaults to 250.
        intensity_thresh_saturation (int, optional): Threshold to check saturation in the patch. Defaults to 5.
        intensity_thresh_b (int, optional): Threshold to check blackness in the patch. Defaults to 128.
        pen_intensity_thresh (int, optional): Threshold to check pen markings in the patch. Defaults to 225.
        pen_intensity_thresh_saturation (int, optional): Threshold to check saturation for pen markings. Defaults to 50.
        pen_intensity_thresh_b (int, optional): Threshold to check blackness for pen markings. Defaults to 128.

    Returns:
        bool: Whether the patch is valid (True) or not (False)
    """
    # Convert the image to HSV color space
    patch_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Calculate the number of white pixels
    white_mask = np.all(img > intensity_thresh, axis=2)
    count_white_pixels = np.sum(white_mask)
    percent_white_pixels = count_white_pixels / (patch_size[0] * patch_size[1])

    # Calculate the number of black pixels
    black_mask = np.all(img < intensity_thresh_b, axis=2)
    count_black_pixels = np.sum(black_mask)
    percent_black_pixels = count_black_pixels / (patch_size[0] * patch_size[1])

    # Calculate the percentage of low saturation pixels
    low_saturation_mask = patch_hsv[..., 1] < intensity_thresh_saturation
    percent_low_saturation = np.sum(low_saturation_mask) / (
        patch_size[0] * patch_size[1]
    )

    # Calculate the percentage of high intensity (brightness) pixels
    high_intensity_mask = patch_hsv[..., 2] > intensity_thresh
    percent_high_intensity = np.sum(high_intensity_mask) / (
        patch_size[0] * patch_size[1]
    )

    # Convert the image to HED color space
    ihc_hed = rgb2hed(img)

    # Rescale intensity of the Hematoxylin channel
    e = rescale_intensity(
        ihc_hed[:, :, 1],
        out_range=(0, 255),
        in_range=(0, np.percentile(ihc_hed[:, :, 1], 99)),
    )

    # Calculate the percentage of low intensity in the rescaled Hematoxylin channel
    percent_low_intensity = np.sum(e < 50) / (patch_size[0] * patch_size[1])

    # Calculate the percentage of low hue values in the HSV image
    percent_low_hue = np.sum(patch_hsv[..., 0] < 128) / (patch_size[0] * patch_size[1])

    # Check the conditions for determining if the patch is invalid
    if (
        percent_low_saturation > 0.99
        or np.mean(patch_hsv[..., 1]) < 5
        or percent_high_intensity > 0.99
    ):
        if percent_low_saturation < 0.1:
            return False
    elif (
        percent_low_saturation > 0.99
        and percent_high_intensity > 0.99
        or percent_black_pixels > 0.99
        or percent_white_pixels > 0.6
        or percent_low_intensity > 0.95
        or percent_low_hue > 0.97
    ):
        return False

    # If none of the conditions for invalidity are met, the patch is valid
    return True

class DsaPatch:
    coordinates: list[float,float]
    file_path: str

    def __init__(self,coordinates,file_path):
        self.coordinates = coordinates
        self.file_path = file_path

def extract_and_save_patches(slide, named_polygons, base_dir, patch_size)->list[DsaPatch]:
    """Extract and save square patches from slide image based on polygons.

    Args:
        slide: Whole slide image.
        named_polygons: List of tuples containing polygon names and polygons.
        base_dir: Path to the base directory where patches will be saved.
        patch_size: Size of the patches to be extracted.

    """
    patches = []
    for name, polygon in named_polygons:
        patch_dir = os.path.join(base_dir, name)
        os.makedirs(patch_dir, exist_ok=True)

        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny

        # Determine the centroid of the polygon to center the patch
        centroid = polygon.centroid
        cx, cy = int(centroid.x), int(centroid.y)

        # Calculate the top left coordinates to extract a centered square patch
        top_left_x = cx - patch_size // 2
        top_left_y = cy - patch_size // 2

        # Ensure the patch does not go outside the slide dimensions
        top_left_x = max(0, min(top_left_x, slide.dimensions[0] - patch_size))
        top_left_y = max(0, min(top_left_y, slide.dimensions[1] - patch_size))

        # Extract the patch
        patch = slide.read_region(
            (top_left_x, top_left_y), 0, (patch_size, patch_size)
        ).convert("RGB")
        patch_image = Image.fromarray(np.array(patch))
        patch_filename = f"patch_{top_left_x}_{top_left_y}.png"
        patch_filepath = os.path.join(patch_dir, patch_filename)
        patch_image.save(patch_filepath)
        dsa_patch = DsaPatch((top_left_x, top_left_y), patch_filepath)
        patches.append(dsa_patch)


        # Setup the overlap size for the next patch
        overlap_size = patch_size // 2

        # If the bounding box is larger than the patch size, continue extracting additional patches
        if width > patch_size or height > patch_size:
            for x in range(int(minx), int(maxx), overlap_size):
                for y in range(int(miny), int(maxy), overlap_size):
                    # Check if the patch is within the polygon before extracting
                    patch_polygon = shapely.geometry.box(
                        x, y, x + patch_size, y + patch_size
                    )
                    if polygon.intersects(patch_polygon):
                        patch = np.array(
                            slide.read_region(
                                (x, y), 0, (patch_size, patch_size)
                            ).convert("RGB")
                        )

                        # Check if the patch is valid with respect to pen markings and artifacts
                        if patch_check(patch):
                            patch_image = Image.fromarray(np.array(patch))
                            patch_filename = f"patch_{x}_{y}.png"
                            patch_filepath = os.path.join(patch_dir, patch_filename)
                            patch_image.save(patch_filepath)
                            dsa_patch = DsaPatch((x,y), patch_filepath)
                            patches.append(dsa_patch)
    return patches

class DsaRowCsv:
    PatchCoordinatesX: float
    PatchCoordinatesY: float
    SlidePatchPath: str
    SubjectID: int

    def __init__(self, PatchCoordinatesX, PatchCoordinatesY,SlidePatchPath, SubjectID):
        self.PatchCoordinatesX = PatchCoordinatesX
        self.PatchCoordinatesY = PatchCoordinatesY
        self.SlidePatchPath = SlidePatchPath
        self.SubjectID = SubjectID




def write_dsa_patches_to_csv(dsa_patches:DsaPatch, subjectID,  output_dir, output_csv):
    csv_filename = os.path.join(output_dir, "list.csv")

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
    new_df_rows = []
    for dsa_patch in dsa_patches:
        patch_coords = dsa_patch.coordinates
        new_row = DsaRowCsv(PatchCoordinatesX=patch_coords[0],PatchCoordinatesY= patch_coords[1], SlidePatchPath=dsa_patch.file_path, SubjectID=subjectID)
        new_df_rows.append(new_row.__dict__)
    new_df = pd.DataFrame(new_df_rows)
    output_df = pd.concat([output_df, new_df])

    output_df.to_csv(csv_filename, index=False)




def extract_patches(json_path, svs_path, patch_size, output_dir,sid):
    """Main function to process the whole slide image and extract patches.

    Args:
        json_path: Path to the JSON annotations file.
        svs_path: Path to the .svs whole slide image file.
        patch_size: Size of the patches to be extracted.
        output_dir: Path to the output directory.

    """
    print(f"Extracting patches from {svs_path} based on {json_path}...")
    annotations = load_annotations(json_path)
    named_polygons = extract_polygons(annotations)
    slide = openslide.OpenSlide(svs_path)
    dsa_patches = extract_and_save_patches(slide, named_polygons, output_dir, patch_size)
    write_dsa_patches_to_csv(dsa_patches,sid, output_dir, output_dir)
