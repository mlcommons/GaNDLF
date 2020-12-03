import math
import sys
from pathlib import Path
import requests, zipfile, io, os

'''
steps to follow to write tests:
[x] download sample data
[ ] construct the training csv
[ ] for each dir (application type) and sub-dir (image dimension), run training for a single epoch on cpu
[ ] for each dir (application type) and sub-dir (image dimension), run inference for a single trained model per testing/validation split for a single subject on cpu
4. hopefully the various sys.exit messages throughout the code will catch issues
'''

def test_download_data():
  '''
  This function downloads the sample data, which is the first step towards getting everything ready
  '''
  urlToDownload = 'https://github.com/sarthakpati/tempDownloads/raw/main/data.zip'
  if not Path(os.getcwd() + '/testing/data/test/3d_rad_segmentation/001/image.nii.gz').exists():
      print('Downloading and extracting sample data')
      r = requests.get(urlToDownload)
      z = zipfile.ZipFile(io.BytesIO(r.content))
      z.extractall('./testing')

def test_full():
  test = 1
  # print('started 2d segmentation')
  # print('passed')
  # print('started 3d segmentation')
  # sys.exit('boo!')
