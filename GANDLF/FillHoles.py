# ------------------------------------------------------------------------------
# Name: Fill Holes
# Date: February 22, 2021
# Version: 1.0
# Author: Jose L. Agraz, PhD
#
# Description: This program fills in holes in a tensor using a given structure.
#              First, the input is eroded and dilated (closed). Then, holes
#              filled in. Keep in mind structure shape and size is key to a
#              successful application. Structure examples in the scipy library
#              are square, rectangle, diamond, disk, cube, octahedron, ball,
#              octagon, and star.
#              See scipy structure documentation "Generate structuring elements"
#
# Input:  Tensor, Structure (optional, default cube, 4x4 pixels)
# Output: Tensor
#
# ------------------------------------------------------------------------------
import torch
from skimage.morphology import cube
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.morphology import binary_closing


def FillHoles(InputTensor, FillingStructure=cube(4)):
    # Convert tensor to numpy array
    InputNumpyArray = InputTensor.numpy()
    # The closing of an input image by erosion and dilation of the image by a structuring element.
    # Closing fills holes smaller than the structuring element
    ClosedArray = binary_closing(InputNumpyArray, structure=FillingStructure).astype(
        int
    )
    # Fill the holes in binary objects
    OutputArray = binary_fill_holes(ClosedArray, structure=FillingStructure).astype(int)
    # convert numpy array to tensor
    TensorOutput = torch.from_numpy(OutputArray)

    return TensorOutput
