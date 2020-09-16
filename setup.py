#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
  readme = readme_file.read()

requirements = [
  'numpy',
  'scipy',
  'SimpleITK==1.2.4',
  'torch>=1.6',
  'torchvision',
  'tqdm',
  'torchio==0.17.34',
  'pandas',
  'pylint',
  'torchsummary',
  'scikit-learn==0.23.1',
  'pickle5==0.0.11',
  'setuptools',
  'pyyaml'
]

setup(
  author="Megh Bhalerao, Siddhesh Thakur, Sarthak Pati",
  author_email='software@cbica.upenn.edu',
  python_requires='>=3.6',
  scripts=['gandlf_run', 'gandlf_constructCSV'],
  classifiers=[
    'Development Status :: Pre-Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD-3-Clause License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
  description=(
    "Semantic segmentation using various DL architectures"
    " on PyTorch."
  ),
  install_requires=requirements,
  license="BSD-3-Clause License",
  long_description=readme,
  long_description_content_type='text/markdown',
  include_package_data=True,
  keywords='semantic, segmentation, brain, breast, liver, lung, augmentation',
  name='GANDLF',
  version='0.0.3.NR', # NR: non-release; this should be changed when tagging
  zip_safe=False,
)
