
import torchio
from torchio.transforms import RandomAffine, RandomElasticDeformation, Compose

class ImagesFromDataFrame():
  """
  Documentation for the class goes here
  """
  def __init__(self, dataFrame, augmentations):
    self.input_df = dataFrame