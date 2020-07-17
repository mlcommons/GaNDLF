
import torchio
from torchio.transforms import RandomAffine, RandomElasticDeformation, Compose

class ImagesFromDataFrame():
  """
  This class takes a pandas dataframe as input and structures it into a data structure that can be passed to Torch
  """
  def __init__(self, dataFrame, augmentations):
    self.input_df = dataFrame