import math
import sys

'''
steps to follow to write tests:
1. download sample data
2. construct the training csv
2. for each dir (application type) and sub-dir (image dimension), run training for a single epoch on cpu
3. for each dir (application type) and sub-dir (image dimension), run inference for a single trained model per testing/validation split for a single subject on cpu
4. hopefully the various sys.exit messages throughout the code will catch issues
'''

def test_first():
  test = 1
  print('started 2d segmentation')
  print('passed')
  print('started 3d segmentation')
  sys.exit('boo!')
